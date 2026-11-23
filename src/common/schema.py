"""Contrato de dados: a ordem das features é a mesma no treino, na API e no simulador."""

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"
