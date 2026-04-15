import random
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from transformer import Transformer
from vocabulary import SUBJECTS, VERBS, NUMBERS, NOUNS


PAD = 0
SOS = 1
EOS = 2
UNK = 3


def _english_verb_form(verb: str, subject_en: str) -> str:
    if subject_en in {"he", "she"}:
        return verb + "s"
    return verb


def generate_example() -> tuple[str, str]:
    subj = random.choice(SUBJECTS)
    verb_en_base, verb_he_forms = random.choice(list(VERBS.items()))
    num_en = random.choice(list(NUMBERS.keys()))
    noun = random.choice(NOUNS)

    verb_en = _english_verb_form(verb_en_base, subj["en"])
    verb_he = verb_he_forms[subj["num"]]
    num_he = NUMBERS[num_en][noun["gender"]]

    if num_en == "one":
        noun_en = noun["en_sg"]
        noun_he = noun["he_sg"]
        tgt = f"{subj['he']} {verb_he} {noun_he} {num_he}"
    else:
        noun_en = noun["en_pl"]
        noun_he = noun["he_pl"]
        tgt = f"{subj['he']} {verb_he} {num_he} {noun_he}"

    src = f"{subj['en']} {verb_en} {num_en} {noun_en}"
    return src, tgt


def generate_dataset(n_samples: int = 1000) -> list[tuple[str, str]]:
    return list(set([generate_example() for _ in range(n_samples)]))


def build_vocab(sentences: list[str]) -> dict[str, int]:
    words = Counter()
    for sent in sentences:
        words.update(sent.split())
    vocab = {
        "<PAD>": PAD,
        "<SOS>": SOS,
        "<EOS>": EOS,
        "<UNK>": UNK,
    }
    for word in sorted(words):
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def encode_sentence(sentence: str, vocab: dict[str, int], add_sos: bool = False, add_eos: bool = True) -> list[int]:
    ids = []
    if add_sos:
        ids.append(SOS)
    ids.extend(vocab.get(tok, UNK) for tok in sentence.split())
    if add_eos:
        ids.append(EOS)
    return ids


def decode_ids(ids: list[int], inv_vocab: dict[int, str]) -> list[str]:
    tokens = []
    for idx in ids:
        token = inv_vocab.get(idx, "<UNK>")
        if token == "<EOS>":
            break
        if token not in {"<PAD>", "<SOS>"}:
            tokens.append(token)
    return tokens


class TranslationDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        src_vocab: dict[str, int],
        tgt_vocab: dict[str, int],
    ):
        self.examples = []
        for src_text, tgt_text in pairs:
            src_ids = encode_sentence(src_text, src_vocab, add_sos=False, add_eos=True)
            tgt_ids = encode_sentence(tgt_text, tgt_vocab, add_sos=True, add_eos=True)
            self.examples.append((src_ids, tgt_ids))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.examples[idx]


def collate_fn(batch: list[tuple[list[int], list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
    src_batch, tgt_batch = zip(*batch)

    max_src_len = max(len(x) for x in src_batch)
    max_tgt_len = max(len(x) for x in tgt_batch)

    padded_src = []
    padded_tgt = []

    for src_ids, tgt_ids in zip(src_batch, tgt_batch):
        padded_src.append(src_ids + [PAD] * (max_src_len - len(src_ids)))
        padded_tgt.append(tgt_ids + [PAD] * (max_tgt_len - len(tgt_ids)))

    return torch.tensor(padded_src, dtype=torch.long), torch.tensor(padded_tgt, dtype=torch.long)


def make_src_mask(src: torch.Tensor) -> torch.Tensor:
    # src: (batch, src_len)
    return (src != PAD).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, src_len)


def make_tgt_mask(tgt: torch.Tensor) -> torch.Tensor:
    # tgt: (batch, tgt_len)
    batch_size, tgt_len = tgt.shape
    tgt_pad_mask = (tgt != PAD).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_len)
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=tgt.device))
    tgt_sub_mask = tgt_sub_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, tgt_len, tgt_len)
    return tgt_pad_mask & tgt_sub_mask  # (batch, 1, tgt_len, tgt_len)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    lr: float = 1e-3,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    train_losses = []
    for epoch in range(num_epochs):
        total_loss = 0.0

        for src, tgt in train_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            # Teacher forcing:
            # decoder input = all target tokens except last
            # prediction target = all target tokens except first
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = make_src_mask(src)
            tgt_mask = make_tgt_mask(tgt_input)

            log_probs = model(src, tgt_input, src_mask, tgt_mask)  # (batch, tgt_len-1, vocab)

            loss = F.nll_loss(
                log_probs.reshape(-1, log_probs.size(-1)),
                tgt_output.reshape(-1),
                ignore_index=PAD,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1:02d} | loss = {avg_loss:.4f}")
    
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("loss.png")


@torch.no_grad()
def translate_sentence(
    model: nn.Module,
    sentence: str,
    src_vocab: dict[str, int],
    tgt_inv_vocab: dict[int, str],
    device: torch.device,
    max_len: int = 12,
) -> str:
    model.eval()

    src_ids = encode_sentence(sentence, src_vocab, add_sos=False, add_eos=True)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    src_mask = make_src_mask(src)

    generated = model.greedy_decode(
        src=src,
        start_token_id=SOS,
        end_token_id=EOS,
        max_len=max_len,
        src_mask=src_mask,
    )

    pred_ids = generated[0].tolist()
    pred_tokens = decode_ids(pred_ids, tgt_inv_vocab)
    return " ".join(pred_tokens)


def main(train_examples: int = 1000, num_epochs: int = 10) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = generate_dataset(int(train_examples * 1.2))
    train_pairs = pairs[:train_examples]

    src_vocab = build_vocab([src for src, _ in train_pairs])
    tgt_vocab = build_vocab([tgt for _, tgt in train_pairs])

    train_loader = DataLoader(
        dataset=TranslationDataset(train_pairs, src_vocab, tgt_vocab),
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = Transformer(
        src_vocab=len(src_vocab),
        tgt_vocab=len(tgt_vocab),
        num_layers=2,
        d_model=128,
        num_heads=4,
        d_ff=256,
        dropout=0.1,
    ).to(device)

    train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        num_epochs=num_epochs,
        lr=1e-3,
    )

    tgt_inv_vocab = {idx: tok for tok, idx in tgt_vocab.items()}

    test_pairs = pairs[train_examples:]
    score = 0
    for src, tgt in test_pairs:
        pred = translate_sentence(model, src, src_vocab, tgt_inv_vocab, device)
        if pred == tgt:
            score += 1
        else:
            print(f"Example failed:\n  src: {src}\n  tgt: {tgt}\n  pred: {pred}\n")
    print(f"\nTest accuracy: {score / len(test_pairs) * 100:.2f}% ({len(test_pairs)} examples)")
    
    print("\nPredictions:")

    test_sentences = [
        "i need six boats",
        "you buy five cars",
        "we request nine computers",
        "i see one cat",
        "we find one boat",
        "you want two dresses",
        "they like two books",
        "he needs three phones",
        "he loves four printers",
    ]

    for sent in test_sentences:
        pred = translate_sentence(model, sent, src_vocab, tgt_inv_vocab, device)
        print(f"{sent} --> {pred}")


if __name__ == "__main__":
    main(train_examples=1000, num_epochs=10)
