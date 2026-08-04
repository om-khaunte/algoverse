# src/embed.py
import torch
import esm
import numpy as np
import pandas as pd
import os

# Load ESM-2 model once
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

def embed_species(df, species_name, batch_size=8):
    """
    df: DataFrame with columns 'protein_id', 'sequence', 'species'
    species_name: which species to filter to and embed
    returns: numpy array of shape (num_proteins, 1280)
    """
    species_df = df[df["species"] == species_name]
    sequences = list(zip(species_df["protein_id"], species_df["sequence"]))

    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        _, _, batch_tokens = batch_converter(batch)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33])
        reps = results["representations"][33]

        for j, (_, seq) in enumerate(batch):
            seq_rep = reps[j, 1 : len(seq) + 1]
            mean_rep = seq_rep.mean(0)
            all_embeddings.append(mean_rep.numpy())

    return np.array(all_embeddings)


def embed_and_save(csv_path, species_name, output_dir="embeddings"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    embeddings = embed_species(df, species_name)
    np.save(f"{output_dir}/{species_name}.npy", embeddings)
    print(f"Saved {len(embeddings)} embeddings for {species_name} to {output_dir}/{species_name}.npy")
    return embeddings
