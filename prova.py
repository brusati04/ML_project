import torch

# Creare un tensore 1D (vettore)
tensor_1d = torch.tensor([1, 2, 3, 4])
print(tensor_1d)

# Creare un tensore 2D (matrice)
tensor_2d = torch.tensor([[1, 2], [3, 4]])
print(tensor_2d)

# Operazioni sui tensori
tensor_sum = tensor_1d + 5  # Somma di ogni elemento con 5
print(tensor_sum)

# Spostare il tensore sulla GPU (se disponibile)
if torch.cuda.is_available():
    tensor_gpu = tensor_1d.to('cuda')
    print(tensor_gpu)
    # otteniamo il print dell'uso effettivo della gpu:
    print(torch.cuda.memory_allocated())
