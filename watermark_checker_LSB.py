import decimal
from collections import deque
import numpy as np
import torch 
from torch import nn
import matplotlib.pyplot as plt

KEY_FILE = "Watermarked_Model_LSB/ID82908069_key"
MODEL_FILE = "Watermarked_Model_LSB/ID82908069_MINST_MODEL_WATERMARKED_LSB_FULLPATTERN_IMAGE.pth"

type_of_watermark = None
watermark_format = None
watermark_size = None
image_size_x = None
image_size_y = None

def float_to_bin(value):
    decimal.getcontext().prec = 6
    
    d = decimal.Decimal(str(abs(value)))
    # print(d)

    sign_bit = '1' if value < 0 else '0'
    
    integer_part = int(d)
    # print(integer_part)
    
    fractional_part = d - integer_part
    # print(fractional_part)
    
    # Converti la parte intera in binario (7 bit)
    integer_binary = format(integer_part, '07b')
    # print(integer_binary)
    
    # Converti la parte frazionaria in binario (24 bit)
    fractional_decimal = int(fractional_part * 1000000)
    fractional_binary = format(fractional_decimal, '024b')
    
    return sign_bit + integer_binary + fractional_binary

def recupera_testo_da_binario(nome_file_binario):
    with open(nome_file_binario, 'rb') as file_binario:
        dati_binari = file_binario.read()
        # print(dati_binari)
        testo_recuperato = dati_binari.decode('utf-8')
        return testo_recuperato
        
def binary_to_image(binary_string, image_shape):
    # Converti la stringa binaria in una lista di interi (0 o 255)
    pixel_values = [255 if bit == '1' else 0 for bit in binary_string]
    
    # Ricostruisci la matrice di pixel
    pixels = np.array(pixel_values, dtype=np.uint8).reshape(image_shape)
    
    # Visualizza l'immagine usando matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(pixels, cmap='gray')
    plt.axis('off')  # Nasconde gli assi
    # plt.title("Immagine Ricostruita")
    plt.show()
    
    return pixels


def binary_to_string(binary_list):
    text = ""
    binary_chars = deque(binary_list)
    
    while binary_chars:
        binary_char = ""
        for _ in range(8):
            if binary_chars:
                binary_char += binary_chars.popleft()
            else:
                break
        
        if len(binary_char) == 8:
            char = chr(int(binary_char, 2))
            text += char
    
    return text


    
class MNISTModelNONLinear(nn.Module):
    def __init__(self,
                 input_layer: int,
                 hidden_layer: int,
                 output_layer: int):        
        super().__init__()
        self.layer_stack = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=input_layer,out_features=hidden_layer),
                nn.ReLU(),
                nn.Linear(in_features=hidden_layer,out_features=output_layer),
                nn.ReLU()
            )
        
    def forward(self, x):
        return self.layer_stack(x)        
     

model = torch.load(MODEL_FILE)


string = recupera_testo_da_binario(KEY_FILE)
array = string.split()
type_of_watermark = int(array[1])
watermark_format = int(array[0])
watermark_size = int(array[2])

if(type_of_watermark == 1):
    image_size_x = int(array[3])
    image_size_y = int(array[4])
    id_start_num = len(bin(int(array[5]))[2:])
    id_end_num = len(bin(int(array[5]))[2:])
    identifier = int(array[5])
else:
    identifier = int(array[3])
    id_start_num = len(bin(int(array[3]))[2:])
    id_end_num = len(bin(int(array[3]))[2:])
    # print(identifier)
if(watermark_format == 0):
    jump = 1
    strt = 0
else:
    if(watermark_format == 1):
        strt = 0
        jump = 2
    else:
        strt = 1
        jump = 2


ris = []
id_start = []
id_end = []
with torch.no_grad():
    for name, param in model.named_parameters():
        if(watermark_size == 0 and id_end_num == 0):
            break
        if 'weight' in name:
            num_rows, num_cols = param.shape
            for r in range(num_rows):
                if(watermark_size == 0 and id_end_num == 0):
                    break
                for c in range(strt,num_cols,jump):
                    numero_formattato = f"{abs( param[r,c]):.7f}"
                    if not numero_formattato.startswith("0.000000"):
                        parametro = param[r,c].clone().detach()
                        copy_rounded = round(parametro.item(),6) #??
                        result = float_to_bin(copy_rounded)
                        if(id_start_num > 0):
                            id_start_num -= 1
                            id_start.append(result[-1])
                        elif(id_start_num == 0 and watermark_size == 0):
                            id_end_num -= 1
                            id_end.append(result[-1])
                        else:
                            ris.append(result[-1])
                            watermark_size -=1
                        if(watermark_size == 0 and id_end_num == 0):
                            break
                        
stringa_unica = ''.join(ris)
id_start = int(''.join(id_start), 2)
id_end = int(''.join(id_end), 2)


if(id_start == id_end == identifier):
    print("the recovered watermark is safe and has not been changed, id: "+str(id_start))
else:
    print("there is a problem with the recovered watermark, it seems corrupt.")
    
if(type_of_watermark == 1):
    binary_to_image(stringa_unica,(image_size_x,image_size_y))
else:
    print(binary_to_string(stringa_unica))