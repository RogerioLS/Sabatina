# Definição de cores para saída formatada
GREEN = "\033[32;1m"
RED = "\033[31;1m"
CYAN = "\033[36;1m"
WHITE = "\033[37;1;4m"
COLOR_LIMITER = "\033[0m"

# Nome do ambiente Conda
ENV_NAME = ds_env

# Lista de dependências
REQUIREMENTS = requirements.txt

# Nome do instalador do Anaconda
CONDA_INSTALLER = Anaconda3-2022.05-Linux-x86_64.sh

# URL de download do Anaconda
CONDA_URL = https://repo.anaconda.com/archive/$(CONDA_INSTALLER)

# Exibe ajuda com os comandos disponíveis
.PHONY: help
help:
	@echo $(CYAN)"📌 Como usar:"$(COLOR_LIMITER)
	@echo $(WHITE)"1️⃣ Instalar o Anaconda (se necessário)"$(COLOR_LIMITER)
	@echo $(GREEN)"   make install_conda"$(COLOR_LIMITER)
	@echo $(WHITE)"2️⃣ Criar e configurar o ambiente Conda"$(COLOR_LIMITER)
	@echo $(GREEN)"   make setup"$(COLOR_LIMITER)
	@echo $(WHITE)"3️⃣ Ativar o ambiente Conda"$(COLOR_LIMITER)
	@echo $(GREEN)"   make activate"$(COLOR_LIMITER)
	@echo $(WHITE)"4️⃣ Salvar as dependências em requirements.txt"$(COLOR_LIMITER)
	@echo $(GREEN)"   make freeze"$(COLOR_LIMITER)
	@echo $(WHITE)"5️⃣ Testar se as bibliotecas estão instaladas corretamente"$(COLOR_LIMITER)
	@echo $(GREEN)"   make test"$(COLOR_LIMITER)
	@echo $(WHITE)"6️⃣ Remover o ambiente Conda"$(COLOR_LIMITER)
	@echo $(GREEN)"   make clean"$(COLOR_LIMITER)
	@echo $(CYAN)"Caso precise de outras libs específicas, basta adicionar no requirements.txt. 🚀"$(COLOR_LIMITER)

# Instala o Anaconda no sistema (Linux)
.PHONY: install_conda
install_conda:
	@echo $(CYAN)"Baixando Anaconda..."$(COLOR_LIMITER)
	wget $(CONDA_URL) -O $(CONDA_INSTALLER)
	@echo $(CYAN)"Instalando Anaconda..."$(COLOR_LIMITER)
	bash $(CONDA_INSTALLER) -b -u -p $$HOME/anaconda3
	@echo $(GREEN)"Anaconda instalado com sucesso!"$(COLOR_LIMITER)
	@echo $(CYAN)"Adicionando Anaconda ao PATH..."$(COLOR_LIMITER)
	echo 'export PATH="$$HOME/anaconda3/bin:$$PATH"' >> $$HOME/.bashrc
	. $$HOME/.bashrc || bash -c "source $$HOME/.bashrc";
	@echo $(CYAN)"Limpando arquivos de instalação..."$(COLOR_LIMITER)
	rm -f $(CONDA_INSTALLER)
	@echo $(GREEN)"Instalador removido! Anaconda pronto para uso. Reinicie o terminal para garantir as mudanças."$(COLOR_LIMITER)

# Cria um ambiente Conda e instala pacotes
.PHONY: setup
setup:
	@echo $(CYAN)"Criando ambiente Conda..."$(COLOR_LIMITER)
	conda create --name $(ENV_NAME) python=3.10 -y
	@echo $(GREEN)"Ambiente Conda criado com sucesso!"$(COLOR_LIMITER)
	@echo $(CYAN)"Instalando pacotes..."$(COLOR_LIMITER)
	conda activate $(ENV_NAME) && pip install --upgrade pip && pip install -r $(REQUIREMENTS)
	@echo $(GREEN)"Pacotes instalados com sucesso!"$(COLOR_LIMITER)

# Ativa o ambiente Conda
.PHONY: activate
activate:
	@echo $(WHITE)"Para ativar o ambiente, execute:"$(COLOR_LIMITER)
	@echo $(CYAN)"conda activate $(ENV_NAME)"$(COLOR_LIMITER)

# Salva as dependências no requirements.txt
.PHONY: freeze
freeze:
	@echo $(CYAN)"Gerando arquivo de dependências..."$(COLOR_LIMITER)
	conda activate $(ENV_NAME) && pip freeze > $(REQUIREMENTS)
	@echo $(GREEN)"Arquivo $(REQUIREMENTS) gerado com sucesso!"$(COLOR_LIMITER)

# Remove o ambiente Conda
.PHONY: clean
clean:
	@echo $(RED)"Removendo ambiente Conda..."$(COLOR_LIMITER)
	conda remove --name $(ENV_NAME) --all -y
	@echo $(GREEN)"Ambiente Conda removido com sucesso!"$(COLOR_LIMITER)

# Testa se as bibliotecas estão instaladas corretamente
.PHONY: test
test:
	@echo $(CYAN)"Testando importação das bibliotecas..."$(COLOR_LIMITER)
	conda activate $(ENV_NAME) && python -c "import numpy, pandas, scipy, sklearn, matplotlib, seaborn, torch, tensorflow; print('All imports successful!')"
	@echo $(GREEN)"Testes concluídos com sucesso!"$(COLOR_LIMITER)
