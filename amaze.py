# O programa que ajuda quem tem labirintite

import scipy.ndimage.morphology as m
import cv2
import numpy as np
import sys
import math
from scipy.stats import mode

BRANCO = 255
PRETO = 0

def redimensionar(image, inicio, fim):
    if(image.shape[0] < 1000):
        return(image, inicio, fim)
    
    pct = 1/(image.shape[1] / 1000)
    
    img=image.copy()
    width = int(img.shape[1] * pct)
    height = int(img.shape[0] * pct)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    inicio = [int(i * pct) for i in inicio]
    fim = [int(i * pct) for i in fim]
    return(img,inicio,fim)
    
    
def encontraBrancos(imagem, ponto,tamanho):
    
    brancos = cv2.findNonZero((imagem).transpose())
    
    distancias = np.sqrt((brancos[:,:,0] - ponto[0]) ** 2 + (brancos[:,:,1] - ponto[1]) ** 2)
    menores = (np.concatenate(distancias)).argsort()[:tamanho]
    
    img = imagem.copy()
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    return brancos[menores]
    
def encontraGrossura(img):
    scannerCol = [img.shape[1] / 2, img.shape[1] / 3, img.shape[1]/1.5, img.shape[1]/1.3]
    scannerLinha = [img.shape[0] / 2, img.shape[0] / 3, img.shape[0]/1.5, img.shape[0]/1.3]
    
    imgcopia = img.copy()
    
    intervalos = []
    for coluna in scannerCol:
        
        intervaloAtual = 0
        
        for pixel in [row[int(coluna)] for row in img]: #acessa coluna
            if(pixel == PRETO):
                intervalos.append(intervaloAtual)
                intervaloAtual=0
            else: 
                intervaloAtual = intervaloAtual + 1
            
        imgcopia[:,int(coluna)] = [125] * len(img[:,int(coluna)] ) #escreve coluna
    
    for linha in scannerLinha:
        
        intervaloAtual = 0
        
        for pixel in img[int(linha)]: #acessa linha
            if(pixel == PRETO):
                intervalos.append(intervaloAtual)
                intervaloAtual=0
            else: 
                intervaloAtual = intervaloAtual + 1
            
        imgcopia[int(linha)] = [125] * len(img[int(linha)] ) #escreve coluna
    
    cv2.imwrite('./output/2_grossura.jpg', imgcopia)
    intervalos = [value for value in intervalos if value != 0]
    
    grossura = math.floor((mode(intervalos).mode - 1)/2)
    return (grossura-1)

def esqueletiza(img):
    h1 = np.array([[0, 0, 0],[0, 1, 0],[1, 1, 1]]) 
    m1 = np.array([[1, 1, 1],[0, 0, 0],[0, 0, 0]]) 
    h2 = np.array([[0, 0, 0],[1, 1, 0],[0, 1, 0]]) 
    m2 = np.array([[0, 1, 1],[0, 0, 1],[0, 0, 0]])
    hit_list = [] 
    miss_list = []
    for k in range(4): 
        hit_list.append(np.rot90(h1, k))
        hit_list.append(np.rot90(h2, k))
        miss_list.append(np.rot90(m1, k))
        miss_list.append(np.rot90(m2, k))    
    img = img.copy()
    while True:
        last = img
        for hit, miss in zip(hit_list, miss_list): 
            hm = m.binary_hit_or_miss(img, hit, miss) 
            img = np.logical_and(img, np.logical_not(hm)) 
        if np.all(img == last):  
            break
    img = img.astype(np.uint8)
    img*=255
    return img    


    
def preprocessamento(maze_image):
    maze_image = cv2.cvtColor(maze_image,cv2.COLOR_BGR2GRAY)
    
    # threshold binario local
    #maze_image = cv2.adaptiveThreshold(maze_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    
    # threshold binario + OTSU
    retval2,maze_image = cv2.threshold(maze_image,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)    
    
    #deteccao de borda
    #maze_image = cv2.Canny(maze_image, 50, 150)
    #maze_image =  cv2.bitwise_not(maze_image)
    return maze_image
    
def amaze(maze_image, inicio, fim, ):
   
    maze_image = cv2.imread(maze_image)
    
    # redimensionar imagem
    maze_image,inicio,fim = redimensionar(maze_image, inicio, fim)
    
    original = maze_image.copy()
    
    # preprocessamento
    maze_image = preprocessamento(maze_image.copy())
    
    cv2.imwrite("./output/1_tratamento.png", maze_image)
        
    
    solucao = None
    
    # tenta com e sem esqueletização
    retry = False
    while((solucao is None) and not retry):
        
        if(retry):
            skel = maze_image.copy()
            retry=True
        else:
            skel = esqueletiza(maze_image.copy())
    
        inicios = encontraBrancos(skel,inicio,15)
        fins = encontraBrancos(skel, fim, 5)
    
        inicial = skel.copy()
        inicial = cv2.cvtColor(skel.copy(),cv2.COLOR_GRAY2RGB)
        for pixel in np.concatenate(inicios):
            inicial[pixel[0],pixel[1]] = [255,0,0]
            
        skeleto = maze_image.copy()
        inicial = cv2.cvtColor(maze_image.copy(),cv2.COLOR_GRAY2RGB) - inicial
        cv2.imwrite("./output/3_skel_"+str(int(retry))+".png", inicial)

        # tenta todos os pontos iniciais
        pontoInicial = 0
        while((solucao is None) & (pontoInicial < len(inicios))):
            
            solucao = bfs(skel, np.concatenate(inicios[pontoInicial]), fins)
           
            if(solucao is not None):
                engrossamento = encontraGrossura(maze_image)
                if(engrossamento > 2):
                    engrossamento = engrossamento - 1
               
                for pixel in solucao:
                     original[pixel[0], pixel[1]] = [15,170,15]
                     for vizinho in vizinhosN(pixel[0],pixel[1],3):
                         if(skeleto[vizinho[0], vizinho[1]] > 125):
                             original[vizinho[0], vizinho[1]] = [15,170,15]

                cv2.imwrite("./output/4_resultado.png", original)
                cv2.imshow("Output", original)
                cv2.waitKey()
            
            pontoInicial = pontoInicial + 1

def bfs(maze_image,inicio,fim):
    fila = []
    fila.append([inicio])
    fim = np.concatenate(fim).tolist()
    
    tamx = maze_image.shape[0]
    tamy = maze_image.shape[1]
    
    while fila:
        
        caminho = fila.pop(0)
        atual = caminho[-1]
        
        if(atual is inicio):
            atual = atual.tolist()
        
        if(atual in fim):
            return caminho
        
        for vizinho in vizinhosDiag(atual[0],atual[1]):
            if(vizinho[0] <= tamy & vizinho[1] <= tamx):
                if(maze_image[vizinho[0],vizinho[1]] == BRANCO):
                    novo_caminho = list(caminho)
                    novo_caminho.append(vizinho)
                    maze_image[vizinho[0],vizinho[1]] = 50
                    
                    fila.append(novo_caminho)

    
def vizinhos(x,y):
    return[[x, y-1],
           [x - 1, y], [x + 1, y],
           [x, y+1]]

def vizinhosDiag(x,y):
    return[[x-1, y-1], [x, y-1], [x+1, y-1],
           [x - 1, y], [x + 1, y],
           [x - 1, y+1], [x, y+1], [x + 1, y+1]]

def vizinhosN(x,y,n):
    vizinhos = []
    for vx in range(x-n, x+n):
        for vy in range(y-n, y+n):
            vizinhos.append([vx,vy])
    return(vizinhos)

    
    
# usage examples
#amaze('alfie.png', [118,307], [269,200])
    
#amaze('maze1.png', [13,156], [320,172])

#amaze('hexagon.png', [28,255], [331,255])

amaze('egip.png', [844,600], [102,509])

#amaze('maze.PNG', [12,178], [198,156])

#amaze('mazePapel.png', [188,159], [800,392])

#amaze('complexo.png',[154,502],[92,476])