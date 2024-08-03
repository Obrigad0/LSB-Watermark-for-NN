import cv2
import decimal
from collections import deque
import random
import numpy as np
import torch 
from torch import nn
import matplotlib.pyplot as plt
import os
import torchvision.models as models
import torchfile

## Choose the type of the watermark
Istext = False # True = text watermark, False = Image watermark
text = "Sapienza Universita' di Roma"
imgPath = "image_example.png"  #The image needs to be black & white!!!
## The payload need to be smaller than the parameters size of the Model.

## Choose the pattern
## MUTUALLY EXCLUSIVE FUNCTIONS
fullPattern = True 
evenPattern = False #False: odd pattern, True: even pattern
 
## 
padding = 0
##


## Now choose the model
resnet = True
MODELFILENAME = 'Watermarked_Model_LSB_Example/CLEAN_cifar10.t7' #Change this to choose the model to place the watermark on
DIRNAME = 'Watermarked_Model_LSB'#not used
modelname = 'MINST_MODEL_WATERMARKED_LSB_' 
identifier = random.randint(10**(8-1), 10**8 - 1)
##



def crea_file_testo(lenbi):
    with open(create_key_name(), 'wb') as file_bin: #
        file_bin.write(key_value(lenbi).encode('utf-8'))


    
def crea_file_binario(nome_file_bin, contenuto):
    # Nome temporaneo per il file di testo
    nome_file_temp = "temp.txt"
    
    # Crea il file di testo temporaneo
    with open(nome_file_temp, 'w') as file_temp:
        file_temp.write(contenuto)
    
    # Converti in binario
    with open(nome_file_temp, 'r') as file_temp:
        testo = file_temp.read()
    
    dati_binari = testo.encode('utf-8')
    
    # Scrivi il file binario
    with open(nome_file_bin, 'wb') as file_bin:
        file_bin.write(dati_binari)
    
    # Elimina il file di testo temporaneo
    os.remove(nome_file_temp)
        


    


def create_key_name():
    return "ID" + str(identifier) + "_key"

def key_value(lenBin): #First value: format, Second value: type of wm, 3 & 4: size 
    key = ""
    if(fullPattern):
        key += "0 "
    elif(evenPattern):
        key += "1 "
    else:
        key += "2 "
    
    if Istext:
        key += "0 " #TEXT
    else:
        key += "1 " #IMAGE
    
    key += str(lenBin)+ " "
    if not Istext:
        key += str(get_image_dimensions(imgPath)[0])+" "
        key += str(get_image_dimensions(imgPath)[1])+" "
    
    key += str(identifier)+" "
    key += str(padding)
    print(key)
    return key
    

def create_model_name(modelname):
    modelname = "ID" + str(identifier) + "_" + modelname
    if(fullPattern):
        modelname = modelname + "FULLPATTERN_"
    elif(evenPattern):
        modelname = modelname + "EVENPATTERN_"
    else:
        modelname = modelname + "ODDPATTERN_"
    
    if Istext:
        modelname += "TEXT"
    else:
        modelname += "IMAGE"
    if resnet:
        modelname += "_resnet"
        modelname += ".t7"
    else:
        modelname += ".pth"
    return modelname


def get_image_dimensions(image_path):
    # Leggi l'immagine
    img = cv2.imread(image_path)
    
    # Ottieni le dimensioni
    if img is not None:
        height, width = img.shape[:2]
        return (width, height)  # Restituisce una tupla
    else:
        return None
    
def binary_to_image(binary_string, image_shape):
    # Converti la stringa binaria in una lista di interi (0 o 255)
    pixel_values = [255 if bit == '1' else 0 for bit in binary_string]
    
    # Ricostruisci la matrice di pixel
    pixels = np.array(pixel_values, dtype=np.uint8).reshape(image_shape)
    
    # Visualizza l'immagine usando matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(pixels, cmap='gray')
    # plt.axis('off')  # Nasconde gli assi
    plt.title("Immagine Ricostruita")
    plt.show()
    
    return pixels

def image_to_binary(image_path):
    # Leggi l'immagine in bianco e nero
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Converti l'immagine in una matrice di pixel
    pixels = np.array(img)
    
    # Converti ogni pixel in un valore binario (0 o 1)
    binary_pixels = (pixels > 127).astype(int)
    
    # Converti la matrice di pixel binari in una lista FIFO
    binary_list = deque(str(pixel) for row in binary_pixels for pixel in row)
    
    return binary_list


def string_to_binary(text):
    binary_list = deque()
    for char in text:
        binary_char = format(ord(char), '08b')
        binary_list.extend(binary_char)
    return binary_list

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

####

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

def bin_to_float(b):
    sign_bit = b[0]
    integer_part = int(b[1:8], 2)
    fractional_part = int(b[8:], 2) / 1000000
    
    value = integer_part + fractional_part
    return -value if sign_bit == '1' else value

def modify_lsb(bin_string, new_lsb):
    return bin_string[:-1] + str(new_lsb)

def operation(num, bit):
    bin_repr = float_to_bin(num)
    modified_bin = modify_lsb(bin_repr, bit)
    return bin_to_float(modified_bin)

def conta_parametri(modello):
    if(not resnet):
        a = sum(p.numel() for name, p in modello.named_parameters() if 'weight' in name)
    else:
        a =  sum(p.numel() for name, p in modello['net'].named_parameters() if 'weight' in name)
    # print(a)
    return a

        



####

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
     
    
        
model = torch.load(MODELFILENAME)     
                

if(Istext):
    binary = string_to_binary(text)
else:
    binary = image_to_binary(imgPath)
lenBin = len(binary)
usable_parameters = conta_parametri(model) - 30 - 16 - padding# For this reason I take away 30, to have a safety margin from these values (read below)
#Some types of values ​​cannot be used to store data such as LSB, 
#these values ​​are values ​​with a very very small decimal number, 
#starting for example from the 7th decimal place ( >= x * 10^-7).
#In a model with 50000 weights the following were found: 1 weight with this characteristic.

if(fullPattern):
    jump = 1
    strt = 0
else:
    padding = int(padding/2)
    if(evenPattern):
        strt = 0
        jump = 2
        usable_parameters = int(usable_parameters / 2)
    else:
        strt = 1
        jump = 2
        usable_parameters = int(usable_parameters / 2)
    
pd = padding

if(usable_parameters < len(binary)):
    
    print("The chosen watermark is too large or there is too much padding, the maximum number of usable bits is: " + str(usable_parameters))
    print("The size of your watermark is:" + str(len(binary)))
    print("The size of the padding is: "+ str(padding))
    
else:
   
            
    id_bin = list(bin(identifier)[2:])
    binary.extend(id_bin)
    for elemento in reversed(id_bin):
        binary.insert(0, elemento)

    if(resnet):
        with torch.no_grad():
            for name, param in model["net"].named_parameters():
                
                if(len(binary) == 0):
                    break
                if 'weight' in name:
                    shape = param.shape
                    if len(shape) == 4:  # Layer convoluzionale 4D
                       for i in range(shape[0]):
                           for j in range(shape[1]):
                               for k in range(shape[2]):
                                   for l in range(strt, shape[3], jump):
                                       if pd > 0:
                                           pd -= 1
                                       else:
                                           numero_formattato = f"{abs(param[i,j,k,l]):.7f}"
                                           if not numero_formattato.startswith("0.000000"):
                                               parametro = param[i,j,k,l].clone().detach()
                                               copy_rounded = round(parametro.item(), 6)
                                               if len(binary) == 0:
                                                   break
                                               result = operation(copy_rounded, binary.popleft())
                                               param[i,j,k,l] = result
                elif len(shape) == 3:  # Layer convoluzionale 3D
                    for i in range(shape[0]):
                        for j in range(shape[1]):
                            for k in range(strt, shape[2], jump):
                                if pd > 0:
                                    pd -= 1
                                else:
                                    numero_formattato = f"{abs(param[i,j,k]):.7f}"
                                    if not numero_formattato.startswith("0.000000"):
                                        parametro = param[i,j,k].clone().detach()
                                        copy_rounded = round(parametro.item(), 6)
                                        if len(binary) == 0:
                                            break
                                        result = operation(copy_rounded, binary.popleft())
                                        param[i,j,k] = result
                elif len(shape) == 2:  # Layer fully connected
                    for r in range(shape[0]):
                        for c in range(strt, shape[1], jump):
                            if pd > 0:
                                pd -= 1
                            else:
                                numero_formattato = f"{abs(param[r,c]):.7f}"
                                if not numero_formattato.startswith("0.000000"):
                                    parametro = param[r,c].clone().detach()
                                    copy_rounded = round(parametro.item(), 6)
                                    if len(binary) == 0:
                                        break
                                    result = operation(copy_rounded, binary.popleft())
                                    param[r,c] = result

    else:

    
        with torch.no_grad():
            for name, param in model.named_parameters():
                
                if(len(binary) == 0):
                    break
                if 'weight' in name:
                    
                    num_rows, num_cols = param.shape
                    for r in range(num_rows):
                        if(len(binary) == 0):
                            break
                        for c in range(strt,num_cols,jump):
                            if(pd > 0):
                                pd -= 1
                            else:                                
                                numero_formattato = f"{abs( param[r,c]):.7f}"
                                if not numero_formattato.startswith("0.000000"):
                                    parametro = param[r,c].clone().detach()
                                    copy_rounded = round(parametro.item(),6)
                                    if(len(binary) == 0):
                                        break
                                    result = operation(copy_rounded, binary.popleft())
                                    param[r,c] = result
        
    model_name = create_model_name(modelname)
    print("Model saved under the name: " + model_name)
    torch.save(model, model_name) #  model_name
    crea_file_testo(lenBin)
    print("Key created")



