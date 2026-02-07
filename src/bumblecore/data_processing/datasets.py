import io

import torch
import torch.distributed as dist
from torch.utils.data import Dataset,get_worker_info
from rich.console import Console
from rich.table import Table
from rich.text import Text
from tqdm import tqdm

import math
import multiprocessing as mp
from itertools import chain


def show_sample(
    input_ids, 
    labels, 
    tokenizer, 
    title="Input and Labels" ,
    left_column = "Input IDs", 
    right_column = "Labels"
):
    input_ids = input_ids.tolist()
    labels = labels.tolist()

    valid_labels_list = [token_id for token_id in labels if token_id != -100]
    decoded_input = tokenizer.decode(input_ids)
    decoded_labels = tokenizer.decode(valid_labels_list)

    table = Table(show_header=True, show_lines=True, title=title)
    table.add_column(left_column, overflow="fold")
    table.add_column(right_column, overflow="fold")

    wrapped_input = Text(decoded_input, no_wrap=False, overflow="fold")
    wrapped_labels = Text(decoded_labels, no_wrap=False, overflow="fold")

    table.add_row(str(input_ids), str(labels))
    table.add_row(wrapped_input, wrapped_labels)

    with io.StringIO() as buf:
        console = Console(file=buf, force_terminal=False)
        console.print(table)
        output = buf.getvalue()

    tqdm.write(output.rstrip())


def get_padding_value(tokenizer):
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    
    eos = tokenizer.eos_token_id
    return eos[0] if isinstance(eos, list) else eos


class PretrainDataset(Dataset):
    
    def __init__(
        self,
        train_dataset,
        tokenizer,
        max_length,
    ):
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.has_shown_sample = False
    
    def __len__(self):
        return len(self.train_dataset)
    
    def create_pretraining_dataset(self, text):

        if self.tokenizer.eos_token_id is not None:
            text = text + self.tokenizer.eos_token
        else:
            text = text + "<|im_end|>"

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False,
        )
        
        input_ids=encoding["input_ids"].squeeze(0)
        attention_mask=encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def _show_train_sample(self, input_ids, labels):

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0 
        worker_info = get_worker_info()
        is_main_worker = (worker_info is None) or (worker_info.id == 0)
        if rank == 0 and is_main_worker and not self.has_shown_sample:
            show_sample(
                input_ids=input_ids,
                labels=labels,
                tokenizer = self.tokenizer,
                title="Pretrain Input and Labels",
                left_column="Input IDs",
                right_column="Labels"
            )
            self.has_shown_sample = True

    def __getitem__(self, idx):
        text = self.train_dataset[idx]["text"]
        sample = self.create_pretraining_dataset(text)

        self._show_train_sample(
            input_ids=sample["input_ids"],
            labels=sample["labels"],
        )

        return sample


def calculate_matched_group(sequences, packing_length: int, is_finished: bool = True):
    """Bin-packing via First Fit Decreasing (https://arxiv.org/pdf/2404.10830)."""
    if len(sequences) == 0:
        return [], []
    import binpacking
    # sequences 是 [(index, length), ...] 列表
    # weight_pos=1 表示长度在元组第二个位置
    # 将一组物品分配到多个容量固定的箱子（bins）中，使得每个箱子的总容量不超过指定的最大值。
    sequences = binpacking.to_constant_volume(sequences, packing_length, weight_pos=1)
    # sequences 是列表的列表，每个子列表包含多个 (index, length) 元组
    # 如果不是最后一批，保留最后一个不完整组用于下一批
    if sequences and not is_finished:
        sequences, ret_sequences = sequences[:-1], sequences[-1]
    else:
        ret_sequences = []
    return sequences, ret_sequences

def split_list(lst, n):
    # 划分列表为n个子列表，对应n个子进程处理
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def _is_master():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _is_dist():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

class SFTDataset(Dataset):
    PACKING_BATCH_SIZE = 1000

    def __init__(
        self,
        train_dataset,
        tokenizer,
        max_length,
        # ── new packing args ──
        packing: bool = False,
        packing_num_proc: int = 1,
    ):
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_shown_sample = False

        # ── packing bookkeeping ──
        self.packing = packing
        self.packing_length = max_length
        self.packed_idx = None
        self.packed_length = None

        if self.packing:
            self.packing_num_proc = min(
                packing_num_proc,
                max(1, math.ceil(len(train_dataset) / self.PACKING_BATCH_SIZE)),
            )
            self._out_queue = mp.Queue()
            self._setup_packing()

    # ------------------------------------------------------------------ #
    #                    packing index construction                       #
    # ------------------------------------------------------------------ #

    def _compute_lengths(self) -> list[int]:
        """Tokenize every sample once to get its length."""
        lengths = []
        for idx in tqdm(range(len(self.train_dataset)), desc="Computing sequence lengths"):
            messages = self.train_dataset[idx]["messages"]
            tools = self.train_dataset[idx].get("tools", None)
            tokens = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                truncation=True,
                max_length=self.max_length,
                tools=tools if tools else None,
            )
            lengths.append(len(tokens))
        return lengths

    def _setup_packing(self):
        """Build packed_idx / packed_length using multi-process bin-packing."""
        if _is_master():
            # 计算每条数据的长度
            lengths = self._compute_lengths()
            offset = 0
            chunked_lengths = split_list(lengths, self.packing_num_proc)

            # launch workers
            for i in range(self.packing_num_proc):
                worker = mp.Process(
                    target=self._create_packed_idx,
                    args=(i, offset, chunked_lengths[i]),
                    daemon=True,
                )
                worker.start()
                offset += len(chunked_lengths[i])

            # collect results
            self.packed_idx = [[] for _ in range(self.packing_num_proc)]
            self.packed_length = [[] for _ in range(self.packing_num_proc)]

            desc = (
                "Packing: "
                if self.packing_num_proc == 1
                else f"Packing (num_proc={self.packing_num_proc}): "
            )
            with tqdm(total=len(lengths), dynamic_ncols=True, desc=desc) as pbar:
                finished = 0
                while finished < self.packing_num_proc:
                    rank, sequences, data_len = self._out_queue.get()
                    if data_len == -1:          # sentinel
                        finished += 1
                        continue
                    pbar.update(data_len)
                    # (idx, length)
                    self.packed_idx[rank] += [[x[0] for x in seq] for seq in sequences]
                    # sum的结果应该接近packing_length
                    self.packed_length[rank] += [sum(x[1] for x in seq) for seq in sequences]

            self.packed_idx = list(chain.from_iterable(self.packed_idx))
            self.packed_length = list(chain.from_iterable(self.packed_length))
        else:
            self.packed_idx, self.packed_length = None, None

        # broadcast to all ranks
        if _is_dist():
            obj_list = [(self.packed_idx, self.packed_length)]
            dist.broadcast_object_list(obj_list)
            self.packed_idx, self.packed_length = obj_list[0]

    def _create_packed_idx(self, rank: int, offset: int, lengths: list[int]):
        """Worker: stream bin-packing results back through self._out_queue."""
        # 这个i + offset 用来定位数据的源数据集中的位置
        data = [(i + offset, length) for i, length in enumerate(lengths)]
        i = 0
        input_data: list = []
        while True:
            new_data = data[i : i + self.PACKING_BATCH_SIZE]
            input_data += new_data
            if not input_data:
                break
            i += self.PACKING_BATCH_SIZE
            is_finished = i >= len(data)
            sequences, input_data = calculate_matched_group(
                input_data, self.packing_length, is_finished=is_finished
            )
            # (进程号，packing结果，剩余数据长度)
            self._out_queue.put((rank, sequences, len(new_data)))
        self._out_queue.put((rank, [], -1))  # sentinel

    # ------------------------------------------------------------------ #
    #                        original SFT logic                          #
    # ------------------------------------------------------------------ #

    def create_conversation_manually(self, messages, tools):

        full = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            truncation=True,
            max_length=self.max_length,
            tools=tools if tools else None,
        )

        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]

        assistant_masks = [0] * len(input_ids)
        current_pos = 0

        for i, message in enumerate(messages):
            if message["role"] == "assistant":
                context_with_reply = messages[: i + 1]
                full_tokens = self.tokenizer.apply_chat_template(
                    context_with_reply,
                    tokenize=True,
                    add_generation_prompt=False,
                    truncation=True,
                    max_length=self.max_length,
                    tools=tools if tools else None,
                )
                reply_end_pos = len(full_tokens)
                assistant_masks[current_pos:reply_end_pos] = [1] * (reply_end_pos - current_pos)
            else:
                if message["role"] == "system":
                    continue
                prompt_context = messages[: i + 1]
                prompt_tokens = self.tokenizer.apply_chat_template(
                    prompt_context,
                    tokenize=True,
                    add_generation_prompt=True,
                    truncation=True,
                    max_length=self.max_length,
                    tools=tools if tools else None,
                )
                current_pos = len(prompt_tokens)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = input_ids.clone()
        labels[torch.tensor(assistant_masks, dtype=torch.bool) == 0] = -100

        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def _show_train_sample(self, input_ids, labels):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        worker_info = get_worker_info()
        is_main_worker = (worker_info is None) or (worker_info.id == 0)
        if rank == 0 and is_main_worker and not self.has_shown_sample:
            show_sample(
                input_ids=input_ids,
                labels=labels,
                tokenizer=self.tokenizer,
                title="SFT Input and Labels",
                left_column="Input IDs",
                right_column="Labels",
            )
            self.has_shown_sample = True

    # ------------------------------------------------------------------ #
    #                      __len__  /  __getitem__                       #
    # ------------------------------------------------------------------ #

    def __len__(self):
        if self.packing:
            return len(self.packed_idx)
        return len(self.train_dataset)

    def _process_single_sample(self, idx: int) -> dict:
        """Tokenize one sample (shared by normal & packing paths)."""
        messages = self.train_dataset[idx]["messages"]
        tools = self.train_dataset[idx].get("tools", None)
        return self.create_conversation_manually(messages, tools)

    def __getitem__(self, idx):
        if self.packing:
            return self._getitem_packing(idx)

        sample = self._process_single_sample(idx)
        self._show_train_sample(input_ids=sample["input_ids"], labels=sample["labels"])
        return sample

    # ── packing __getitem__ ──────────────────────────────────────────────

    def _getitem_packing(self, idx):
        """
        Concatenate the samples assigned to this pack, add per-sequence
        position_ids (reset to 0 at each boundary), and pad to
        ``packing_length`` so every item in the batch has the same shape.

        Returns
        -------
        dict with keys: input_ids, attention_mask, labels, position_ids
            All of shape ``(packing_length,)``.

        Notes
        -----
        * ``position_ids`` resets to 0 at each sequence boundary, which
          Flash-Attention-2 / flex-attention can use to build a
          block-diagonal mask automatically.
        * Padding tokens get ``label = -100``, ``attention_mask = 0``,
          ``position_ids = 0``.
        """
        sequence_indices = self.packed_idx[idx]

        all_input_ids = []
        all_labels = []
        all_position_ids = []

        for seq_idx in sequence_indices:
            sample = self._process_single_sample(seq_idx)
            input_ids = sample["input_ids"]   # (seq_len,)
            labels = sample["labels"]         # (seq_len,)
            seq_len = input_ids.size(0)

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_position_ids.append(torch.arange(seq_len, dtype=torch.long))

        # concat
        input_ids = torch.cat(all_input_ids, dim=0)
        labels = torch.cat(all_labels, dim=0)
        position_ids = torch.cat(all_position_ids, dim=0)
        attention_mask = torch.ones(input_ids.size(0), dtype=torch.long)

        # pad to packing_length
        total_len = input_ids.size(0)
        if total_len < self.packing_length:
            pad_len = self.packing_length - total_len
            pad_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else 0
            )
            input_ids = torch.cat(
                [input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)]
            )
            labels = torch.cat(
                [labels, torch.full((pad_len,), -100, dtype=torch.long)]
            )
            position_ids = torch.cat(
                [position_ids, torch.zeros(pad_len, dtype=torch.long)]
            )
            attention_mask = torch.cat(
                [attention_mask, torch.zeros(pad_len, dtype=torch.long)]
            )

        self._show_train_sample(input_ids=input_ids, labels=labels)

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )


class DPODataset(Dataset):
    def __init__(self, train_dataset, tokenizer, max_length):
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.has_shown_sample = False
    
    def __len__(self):
        return len(self.train_dataset)

    def create_dpo_dataset(self, messages, tools):
        full = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            truncation=True,
            max_length=self.max_length,
            tools=tools if tools else None,
        )

        prompt_messages = messages[:-1]
        prompt_input_ids = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            max_length=self.max_length,
            tools=tools if tools else None,
        )
        input_ids = torch.tensor(full["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(full["attention_mask"], dtype=torch.long)
        prompt_len = len(prompt_input_ids)
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    
    def _show_train_sample(
            self, 
            chosen_input_ids, 
            chosen_labels, 
            rejected_input_ids, 
            rejected_labels, 
        ):

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0 
        worker_info = get_worker_info()
        is_main_worker = (worker_info is None) or (worker_info.id == 0)
        if rank == 0 and is_main_worker and not self.has_shown_sample:
            show_sample(
                input_ids=chosen_input_ids,
                labels=chosen_labels,
                tokenizer = self.tokenizer,
                title="DPO Chosen Input IDs and Labels",
                left_column="Chosen Input IDs",
                right_column="Chosen Labels"
            )
            show_sample(
                input_ids=rejected_input_ids,
                labels=rejected_labels,
                tokenizer = self.tokenizer,
                title="DPO Rejected Input IDs and Labels",
                left_column="Rejected Input IDs",
                right_column="Rejected Labels"
            )
            self.has_shown_sample = True
    
    def __getitem__(self, idx):
        chosen_messages = self.train_dataset[idx]["chosen_messages"]
        rejected_messages = self.train_dataset[idx]["rejected_messages"]

        chosen_data = self.create_dpo_dataset(chosen_messages["messages"], chosen_messages["tools"])
        rejected_data = self.create_dpo_dataset(rejected_messages["messages"], rejected_messages["tools"])

        self._show_train_sample(
            chosen_input_ids=chosen_data["input_ids"],
            chosen_labels=chosen_data["labels"],
            rejected_input_ids=rejected_data["input_ids"],
            rejected_labels=rejected_data["labels"],
        )

        return dict(
            chosen_input_ids=chosen_data["input_ids"],
            chosen_attention_mask=chosen_data["attention_mask"],
            chosen_labels=chosen_data["labels"],
            rejected_input_ids=rejected_data["input_ids"],
            rejected_attention_mask=rejected_data["attention_mask"],
            rejected_labels=rejected_data["labels"],
        )


class DataCollator:
    def __init__(self, tokenizer):
        self.input_ids_padding_value = get_padding_value(tokenizer=tokenizer)

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        input_ids = self.right_pad_sequence(
            input_ids, padding_value=self.input_ids_padding_value
        )
        attention_mask = self.right_pad_sequence(attention_mask, padding_value=0)
        labels = self.right_pad_sequence(labels, padding_value=-100)

        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    @staticmethod
    def right_pad_sequence(sequences, padding_value):
        padded = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=padding_value
        )
        return padded


class DPOCollator:
    def __init__(self, tokenizer):
        self.input_ids_padding_value = get_padding_value(tokenizer=tokenizer)

    def __call__(self, batch):
        chosen_input_ids = [item["chosen_input_ids"] for item in batch]
        chosen_attention_mask = [item["chosen_attention_mask"] for item in batch]
        chosen_labels = [item["chosen_labels"] for item in batch]

        rejected_input_ids = [item["rejected_input_ids"] for item in batch]
        rejected_attention_mask = [item["rejected_attention_mask"] for item in batch]
        rejected_labels = [item["rejected_labels"] for item in batch]

        all_lengths = [
            len(x) for x in chosen_input_ids + rejected_input_ids
        ]
        max_length = max(all_lengths)

        return dict(
            chosen_input_ids=self._right_pad_to_len(
            chosen_input_ids, max_length, self.input_ids_padding_value
        ),
            chosen_attention_mask=self._right_pad_to_len(
            chosen_attention_mask, max_length, 0
        ),
            chosen_labels=self._right_pad_to_len(
            chosen_labels, max_length, -100
        ),
            rejected_input_ids=self._right_pad_to_len(
            rejected_input_ids, max_length, self.input_ids_padding_value
        ),
            rejected_attention_mask=self._right_pad_to_len(
            rejected_attention_mask, max_length, 0
        ),
            rejected_labels=self._right_pad_to_len(
            rejected_labels, max_length, -100
        ),
        )

    @staticmethod
    def _right_pad_to_len(sequences, max_length, padding_value):
        padded = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=padding_value
        )
        if padded.size(1) < max_length:
            diff = max_length - padded.size(1)
            pad_tensor = torch.full(
                (padded.size(0), diff),
                padding_value,
                dtype=padded.dtype,
                device=padded.device
            )
            padded = torch.cat([padded, pad_tensor], dim=1)
        return padded
