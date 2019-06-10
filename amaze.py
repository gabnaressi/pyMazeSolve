# O programa que ajuda quem tem labirintite

import cv2
import numpy as np
import sys
import math
from scipy.stats import mode

BRANCO = 255
PRETO = 0

def redimensionar(image, inicio, fim):
    if(image.shape[0] < 1200):
        return(image, inicio, fim)
    
    pct = 1/(image.shape[1] / 300)
    
    img=image.copy()
    width = int(img.shape[1] * pct)
    height = int(img.shape[0] * pct)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    inicio = [int(i * pct) for i in inicio]
    fim = [int(i * pct) for i in fim]
    return(img,inicio,fim)
    
    
def encontraBrancos(imagem, ponto,tamanho,maxdist):
    
    brancos = cv2.findNonZero((imagem).transpose())
    
    distancias = np.sqrt((brancos[:,:,0] - ponto[0]) ** 2 + (brancos[:,:,1] - ponto[1]) ** 2)
    menores = (np.concatenate(distancias)).argsort()[:tamanho]
    
    img = imagem.copy()
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    return brancos[menores]
  
def esqueletizarLabirinto(maze_image, n):
    
    if(n==0):
        n=1/3    
    
    elemento = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    maze_image = cv2.erode(maze_image, elemento, iterations=int(n*3))#######
    maze = maze_image.copy()
    tamanho = np.size(maze)
    skel = np.zeros(maze.shape,np.uint8)
    
    
    done = False
    
    while (not done):
        org = skel.copy()
        eroded = cv2.erode(maze, elemento)
        temp = cv2.dilate(eroded,elemento)
        temp = cv2.subtract(maze, temp)
        skel = cv2.bitwise_or(skel, temp)
        maze = eroded.copy()
        
        zeros = tamanho - cv2.countNonZero(maze)
        if(zeros == tamanho):
            done=True
        
    if(n==1/3):
        n=1/2
    skel = cv2.dilate(skel,elemento,iterations=int(n*2))#########
    
    return skel
    
def antigoEsqueletizarLabirinto(maze_image,n):
    return cv2.erode(maze_image, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=n)

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
        
        for pixel in [row[int(linha)] for row in img]: #acessa linha
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
    
def amaze(maze_image, inicio, fim):
    
    maze_image = cv2.imread(maze_image)
    
    # redimensionar imagem
    maze_image,inicio,fim = redimensionar(maze_image, inicio, fim)
    
    original = maze_image.copy()
    
    maze_image = cv2.cvtColor(maze_image,cv2.COLOR_BGR2GRAY)
    
    # threshold binario local
    #maze_image = cv2.adaptiveThreshold(maze_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    
    # threshold binario + OTSU
    retval2,maze_image = cv2.threshold(maze_image,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)    
    
    #deteccao de borda
    #maze_image = cv2.Canny(maze_image, 50, 150)
    #maze_image =  cv2.bitwise_not(maze_image)
    
    cv2.imwrite("./output/1_tratamento.png", maze_image)
        
    
    solucao = None
    
    # tenta esqueletizações diferentes
    fator = encontraGrossura(maze_image.copy())
    while((solucao is None) and fator >= 0):
        
        if(fator==0):
            skel = maze_image.copy()
        else:
            skel = antigoEsqueletizarLabirinto(maze_image.copy(),fator)
           
        inicios = encontraBrancos(skel,inicio,15,fator)
        fins = encontraBrancos(skel, fim, 15,fator)
    
        
        inicial = skel.copy()
        inicial = cv2.cvtColor(skel.copy(),cv2.COLOR_GRAY2RGB)
        for pixel in np.concatenate(inicios):
            inicial[pixel[0],pixel[1]] = [255,0,0]
            
        
        inicial = cv2.cvtColor(maze_image.copy(),cv2.COLOR_GRAY2RGB) - inicial
        cv2.imwrite("./output/3_skel_"+str(fator)+".png", inicial)

        # tenta todos os pontos iniciais
        pontoInicial = 0
        while((solucao is None) & (pontoInicial < len(inicios))):
            skeleto = skel.copy()
            solucao = bfs(skel, np.concatenate(inicios[pontoInicial]), fins)
            

            
            if(solucao is not None):
                for pixel in solucao:
                     original[pixel[0], pixel[1]] = [255,0,0]
                     for vizinho in vizinhosN(pixel[0],pixel[1],2):
                         if(skeleto[vizinho[0], vizinho[1]] > 125):
                             original[vizinho[0], vizinho[1]] = [255,0,0]

                cv2.imwrite("./output/4_resultado.png", original)
                cv2.imshow("Output", original)
                cv2.waitKey()
            
            pontoInicial = pontoInicial + 1
    
        fator = fator - 1

def bfs(maze_image,inicio,fim):
    fila = []
    fila.append([inicio])
    fim = np.concatenate(fim).tolist()
    
    while fila:
        
        caminho = fila.pop(0)
        atual = caminho[-1]
        
        if(atual is inicio):
            atual = atual.tolist()
        
        if(atual in fim):
            return caminho
        
        for vizinho in vizinhos(atual[0],atual[1]):
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
amaze('alfie.png', [118,307], [269,200])
    
#amaze(cv2.imread('maze1.png'), [13,156], [320,172])

#amaze(cv2.imread('maze.PNG'), [12,178], [198,156])

#amaze(cv2.imread('mazePapel.png'), [188,159], [800,392])