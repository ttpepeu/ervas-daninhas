import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import os
from datetime import datetime

# Cria as pastas se não existirem
os.makedirs('vegetacao/milho', exist_ok=True)
os.makedirs('vegetacao/buva', exist_ok=True)

# Configurações de validação
CONFIANCA_MINIMA = 0.3  # 30% de confiança mínima
LUMINOSIDADE_MINIMA = 40  # valor mínimo de luminosidade média
LUMINOSIDADE_MAXIMA = 240  # valor máximo de luminosidade média
NITIDEZ_MINIMA = 50  # valor mínimo do laplaciano para considerar imagem nítida
TAMANHO_MINIMO = 224  # tamanho mínimo em pixels

# Cores para visualização (BGR)
COR_MILHO = (0, 255, 0)  # Verde
COR_BUVA = (0, 0, 255)   # Vermelho
COR_TEXTO = (255, 255, 255)  # Branco
COR_AVISO = (0, 165, 255)    # Laranja

def verificar_luminosidade(frame):
    # Verifica se a luminosidade da imagem está adequada
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    luminosidade_media = np.mean(hsv[:, :, 2])
    return LUMINOSIDADE_MINIMA <= luminosidade_media <= LUMINOSIDADE_MAXIMA, luminosidade_media

def verificar_nitidez(frame):
    # Verifica se a imagem está nítida usando o Laplaciano
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nitidez = cv2.Laplacian(gray, cv2.CV_64F).var()
    return nitidez > NITIDEZ_MINIMA, nitidez

def verificar_tamanho(frame):
    # Verifica se a imagem tem tamanho mínimo adequado
    altura, largura = frame.shape[:2]
    return altura >= TAMANHO_MINIMO and largura >= TAMANHO_MINIMO

def validar_imagem(frame):
    # Realiza todas as validações na imagem
    if not verificar_tamanho(frame):
        return False, "Imagem muito pequena"
    
    nitidez_ok, valor_nitidez = verificar_nitidez(frame)
    if not nitidez_ok:
        return False, f"Imagem desfocada (nitidez: {valor_nitidez:.1f})"
    
    luminosidade_ok, valor_luminosidade = verificar_luminosidade(frame)
    if not luminosidade_ok:
        return False, f"Luminosidade inadequada ({valor_luminosidade:.1f})"
    
    return True, "Imagem válida"

def desenhar_classificacao(frame, tipo_planta, confianca):
    # Desenha uma borda colorida e texto indicando a classificação
    altura, largura = frame.shape[:2]
    espessura = 2
    cor = COR_MILHO if tipo_planta == 'milho' else COR_BUVA
    
    # Desenha borda
    cv2.rectangle(frame, (0, 0), (largura, altura), cor, espessura)
    
    # Prepara o texto
    texto = f"{tipo_planta.upper()}: {confianca*100:.1f}%"
    
    # Desenha fundo para o texto
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala = 1.5
    espessura_texto = 2
    (largura_texto, altura_texto), _ = cv2.getTextSize(texto, fonte, escala, espessura_texto)
    
    # Posição do texto centralizado no topo
    pos_x = (largura - largura_texto) // 2
    pos_y = altura_texto + 20
    
    # Desenha retângulo de fundo
    padding = 10
    cv2.rectangle(frame, 
                 (pos_x - padding, pos_y - altura_texto - padding),
                 (pos_x + largura_texto + padding, pos_y + padding),
                 cor, -1)
    
    # Desenha o texto
    cv2.putText(frame, texto, (pos_x, pos_y),
                fonte, escala, COR_TEXTO, espessura_texto)

def carregar_modelo():
    # Carrega o modelo pré-treinado MobileNetV2
    return MobileNetV2(weights='imagenet')

def processar_frame(frame, tamanho_alvo=(224, 224)):
    # Processa o frame para predição do modelo
    # Converte BGR para RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Redimensiona o frame
    redimensionado = cv2.resize(frame_rgb, tamanho_alvo)
    
    # Converte para array e pré-processa
    array_imagem = image.img_to_array(redimensionado)
    array_imagem = np.expand_dims(array_imagem, axis=0)
    frame_processado = preprocess_input(array_imagem)
    
    return frame_processado

def salvar_imagem(frame, tipo_planta, confianca):
    # Salva a imagem na pasta correspondente
    # Primeiro valida a imagem
    valida, mensagem = validar_imagem(frame)
    if not valida:
        print(f"Imagem não salva: {mensagem}")
        return False
    
    # Se a confiança for menor que o mínimo, não salva
    if confianca < CONFIANCA_MINIMA * 100:
        print(f"Confiança muito baixa: {confianca:.1f}%")
        return False
    
    # Cria nome do arquivo com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"vegetacao/{tipo_planta}/{tipo_planta}_{timestamp}_{int(confianca)}pct.jpg"
    
    # Salva a imagem
    cv2.imwrite(nome_arquivo, frame)
    print(f"Imagem salva: {nome_arquivo}")
    return True

def main():
    # Carrega o modelo
    print("Carregando modelo...")
    modelo = carregar_modelo()
    print("Modelo carregado com sucesso!")
    print(f"Configurações de validação:")
    print(f"- Confiança mínima: {CONFIANCA_MINIMA*100}%")
    print(f"- Luminosidade: {LUMINOSIDADE_MINIMA}-{LUMINOSIDADE_MAXIMA}")
    print(f"- Nitidez mínima: {NITIDEZ_MINIMA}")
    print(f"- Tamanho mínimo: {TAMANHO_MINIMO}x{TAMANHO_MINIMO} pixels")

    # Inicializa a webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera")
        return

    print("Iniciando captura de vídeo... Pressione 'q' para sair")
    print("As imagens detectadas serão salvas na pasta 'vegetacao'")

    # Controle para não salvar muitas imagens seguidas da mesma planta
    ultimo_salvamento = {'milho': datetime.now(), 'buva': datetime.now()}
    intervalo_minimo = 2  # segundos entre salvamentos

    while True:
        # Lê o frame da webcam
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível ler o frame")
            break

        # Cria uma cópia do frame para desenhar
        frame_display = frame.copy()

        # Processa o frame
        frame_processado = processar_frame(frame)
        
        # Faz a predição
        predicoes = modelo.predict(frame_processado)
        predicoes_decodificadas = decode_predictions(predicoes, top=3)[0]
        
        # Verifica se há milho ou buva nas principais predições
        agora = datetime.now()
        melhor_predicao = None
        maior_confianca = 0

        # Realiza a validação da imagem e mostra o status
        valida, mensagem = validar_imagem(frame)
        
        for _, rotulo, confianca in predicoes_decodificadas:
            if 'corn' in rotulo.lower() and confianca > maior_confianca:
                melhor_predicao = ('milho', confianca)
                maior_confianca = confianca
                if valida and (agora - ultimo_salvamento['milho']).total_seconds() > intervalo_minimo:
                    if salvar_imagem(frame, 'milho', confianca * 100):
                        ultimo_salvamento['milho'] = agora
            elif ('weed' in rotulo.lower() or 'plant' in rotulo.lower()) and confianca > maior_confianca:
                melhor_predicao = ('buva', confianca)
                maior_confianca = confianca
                if valida and (agora - ultimo_salvamento['buva']).total_seconds() > intervalo_minimo:
                    if salvar_imagem(frame, 'buva', confianca * 100):
                        ultimo_salvamento['buva'] = agora

        # Desenha a classificação se houver uma predição válida
        if melhor_predicao and melhor_predicao[1] >= CONFIANCA_MINIMA:
            desenhar_classificacao(frame_display, melhor_predicao[0], melhor_predicao[1])
        
        # Mostra o status da validação
        if not valida:
            cv2.putText(frame_display, f"Status: {mensagem}", (10, frame_display.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COR_AVISO, 2)

        # Mostra o frame
        cv2.imshow('Detector de Plantas', frame_display)
        
        # Encerra o loop ao pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera os recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
