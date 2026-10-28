# MoveSens — Reconhecimento de Gestos com Python

Projeto de reconhecimento de gestos com as mãos utilizando Python, OpenCV, MediaPipe e Machine Learning.

O MoveSens utiliza a webcam para identificar diferentes gestos e exibir a imagem correspondente ao gesto reconhecido.

## Gestos reconhecidos

- 🤘 Rock
- 🤙 Hang Loose
- ✌️ Paz
- 👍 Joia
- ❤️ Coração
- 👎 Negativo

## Tecnologias utilizadas

- Python
- OpenCV
- MediaPipe
- Pandas
- Scikit-learn
- Joblib

## Como funciona?

1. A webcam captura a imagem da mão.
2. O MediaPipe identifica 21 pontos de referência da mão.
3. As coordenadas X e Y desses pontos são utilizadas como características.
4. Um modelo de Machine Learning classifica o gesto.
5. O sistema exibe a imagem correspondente ao gesto identificado.

Cada gesto é representado por 42 características: duas coordenadas para cada um dos 21 pontos da mão.

## Estrutura do projeto

```text
MoveSens/
├── dados/
├── imagens/
├── coleta_dados.py
├── treino_modelo.py
├── sensor.py
├── requirements.txt
└── README.md
```

## Instalação

É necessário ter o Python 3.11 instalado e uma webcam disponível.

Clone o repositório:

```bash
git clone URL_DO_REPOSITORIO
```

Entre na pasta:

```bash
cd MoveSens
```

Crie um ambiente virtual:

```bash
python -m venv venv
```

Ative o ambiente virtual no Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

## Como executar

### 1. Coletar os dados

```bash
python coleta_dados.py
```

O programa permite capturar amostras dos gestos para criar os arquivos de dados utilizados no treinamento.

### 2. Treinar o modelo

```bash
python treino_modelo.py
```

Após a coleta dos dados, esse script treina o classificador e gera o arquivo `modelo_gestos.pkl`.

### 3. Executar o reconhecimento

```bash
python sensor.py
```

O programa utiliza a webcam para reconhecer os gestos em tempo real.

## Resultados

Na avaliação realizada durante o desenvolvimento, o modelo alcançou aproximadamente **99,82% de acurácia**.

Esse resultado corresponde à divisão de dados utilizada no teste e não garante o mesmo desempenho em situações reais. Como as amostras foram coletadas em sequência, testes com novas gravações e condições diferentes são importantes para avaliar a capacidade de generalização do modelo.

## Autor

**Misael Souza Braga Junior**

GitHub: [MisaelJunior-web](https://github.com/MisaelJunior-web)