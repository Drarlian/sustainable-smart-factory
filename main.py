"""
Fábrica Inteligente Sustentável — Exercício Integrador (NumPy, Pandas, Matplotlib, Scikit‑learn, TensorFlow)

O projeto também pode ser encontrado no gitgub: https://github.com/Drarlian/sustainable-smart-factory.git

O script foi escrito para ser didático e robusto:
- Precisa do arquivo fabrica_energia.csv, no mesmo diretório do script para funcionar.
- Gera um relatório em texto com estatísticas básicas.
- Cria gráficos (dispersão e linha) e salva em PNG.
- Treina um classificador de Árvore de Decisão para "alto_consumo".
- Treina uma rede neural simples em TensorFlow para prever consumo_kwh.

Estrutura esperada do CSV (cabeçalhos):
    dia, horas_trabalhadas, unidades_produzidas, consumo_kwh, maquina
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

import tensorflow as tf


def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Lê o CSV de produção/energia.
    Aplica validações simples e tipos adequados.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normaliza nomes de colunas (caso venham com espaços ou maiúsculas)
    df.columns = [c.strip().lower() for c in df.columns]

    expected = {"dia", "horas_trabalhadas", "unidades_produzidas", "consumo_kwh", "maquina"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing} — esperado: {sorted(expected)}")

    # Garante tipos numéricos onde faz sentido
    numeric_cols = ["dia", "horas_trabalhadas", "unidades_produzidas", "consumo_kwh"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=numeric_cols)

    # Ordena por dia para gráficos de série temporal
    df = df.sort_values("dia").reset_index(drop=True)
    return df


def compute_basic_stats(df: pd.DataFrame) -> dict:
    """
    Manipulação de Dados com Pandas + NumPy.
    Calcula média, desvio padrão e total de energia consumida, além de menor/maior consumo e respectivos dias.
    Retorna dicionário com resultados — também útil para exportar como JSON.
    """
    stats = {
        "media_consumo": float(df["consumo_kwh"].mean()),
        "desvio_padrao_consumo": float(df["consumo_kwh"].std(ddof=1)),
        "total_consumo": float(df["consumo_kwh"].sum()),
        "dia_maior_consumo": int(df.loc[df["consumo_kwh"].idxmax(), "dia"]),
        "valor_maior_consumo": float(df["consumo_kwh"].max()),
        "dia_menor_consumo": int(df.loc[df["consumo_kwh"].idxmin(), "dia"]),
        "valor_menor_consumo": float(df["consumo_kwh"].min()),
        "n_registros": int(len(df)),
    }
    return stats


def save_report(stats: dict, outdir: Path) -> Path:
    """
    Gera um relatório simples em texto (.md) com as estatísticas principais.
    """
    report_path = outdir / "relatorio_estatistico.md"
    lines = [
        "# Relatório — Fábrica Inteligente Sustentável",
        "",
        "## Estatísticas de Consumo (kWh)",
        f"- Registros analisados: **{stats['n_registros']}**",
        f"- Média de consumo: **{stats['media_consumo']:.2f} kWh**",
        f"- Desvio padrão: **{stats['desvio_padrao_consumo']:.2f} kWh**",
        f"- Total consumido: **{stats['total_consumo']:.2f} kWh**",
        f"- Maior consumo: **{stats['valor_maior_consumo']:.2f} kWh** no dia **{stats['dia_maior_consumo']}**",
        f"- Menor consumo: **{stats['valor_menor_consumo']:.2f} kWh** no dia **{stats['dia_menor_consumo']}**",
        "",
        "Gerado automaticamente por `fabrica_inteligente_sustentavel.py`.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    # Salvando em JSON também, para automações:
    (outdir / "estatisticas.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def plot_scatter(df: pd.DataFrame, outdir: Path) -> Path:
    """
    Visualização com Matplotlib.
    Gráfico de dispersão: horas_trabalhadas × consumo_kwh.
    """
    fig, ax = plt.subplots()
    ax.scatter(df["horas_trabalhadas"], df["consumo_kwh"])
    ax.set_title("Dispersão — Horas Trabalhadas × Consumo (kWh)")
    ax.set_xlabel("Horas trabalhadas (dia)")
    ax.set_ylabel("Consumo (kWh)")
    ax.grid(True, linestyle="--", alpha=0.4)
    path = outdir / "dispersao_horas_vs_consumo.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_line_consumo(df: pd.DataFrame, outdir: Path) -> Path:
    """
    Visualização com Matplotlib.
    Gráfico de linha do consumo ao longo dos dias.
    """
    fig, ax = plt.subplots()
    ax.plot(df["dia"], df["consumo_kwh"], marker="o")
    ax.set_title("Consumo de Energia ao Longo dos Dias")
    ax.set_xlabel("Dia")
    ax.set_ylabel("Consumo (kWh)")
    ax.grid(True, linestyle="--", alpha=0.4)
    path = outdir / "linha_consumo_dias.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def add_alto_consumo_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classificação com Scikit‑learn
    Adiciona coluna binária 'alto_consumo' (1 se consumo > média, 0 caso contrário).
    """
    media = df["consumo_kwh"].mean()
    df = df.copy()
    df["alto_consumo"] = (df["consumo_kwh"] > media).astype(int)
    return df


def train_decision_tree(df: pd.DataFrame, outdir: Path) -> Tuple[DecisionTreeClassifier, float, Path]:
    """
    Treina um DecisionTreeClassifier para prever 'alto_consumo' a partir de:
    - horas_trabalhadas
    - unidades_produzidas

    Retorna o modelo, acurácia de teste e o caminho da matriz de confusão (PNG).
    """
    X = df[["horas_trabalhadas", "unidades_produzidas"]].values
    y = df["alto_consumo"].values

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    # Árvore simples (parâmetros conservadores pelo tamanho do dataset)
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    # Relatório textual
    (outdir / "arvore_relatorio.txt").write_text(
        f"Acurácia: {acc:.3f}\n\n{classification_report(y_test, y_pred)}", encoding="utf-8"
    )

    # Matriz de confusão
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    ax.set_title("Matriz de Confusão — Árvore de Decisão (alto_consumo)")
    path = outdir / "arvore_confusao.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

    return clf, acc, path


def train_tensorflow_regressor(df: pd.DataFrame, outdir: Path):
    """
    Predição com TensorFlow.
    Treina uma rede neural densamente conectada para prever consumo_kwh a partir de
    ['horas_trabalhadas', 'unidades_produzidas'].
    """
    features = df[["horas_trabalhadas", "unidades_produzidas"]].astype(float).values
    target = df["consumo_kwh"].astype(float).values

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Escalonamento simples ajuda redes neurais
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Modelo sequencial simples
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(x_train_scaled.shape[1],)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1)  # saída contínua (regressão)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    history = model.fit(x_train_scaled, y_train, validation_split=0.2, epochs=200, batch_size=8, verbose=0)

    eval_loss, eval_mae = model.evaluate(x_test_scaled, y_test, verbose=0)

    # Curva de aprendizado
    fig, ax = plt.subplots()
    ax.plot(history.history["loss"], label="train_loss")
    ax.plot(history.history["val_loss"], label="val_loss")
    ax.set_title("Curva de Treinamento — Rede Neural (MSE)")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.legend()
    curve_path = outdir / "tf_curva_treinamento.png"
    fig.tight_layout()
    fig.savefig(curve_path, dpi=140)
    plt.close(fig)

    # Salva métricas e modelo
    metrics_path = outdir / "tf_metrics.json"
    metrics_path.write_text(json.dumps({"mse": float(eval_loss), "mae": float(eval_mae)}, indent=2), encoding="utf-8")

    model_path = outdir / "tf_model.keras"
    model.save(model_path)

    # Salvando o scaler para uso posterior
    import joblib  # scikit-learn dependency
    joblib.dump(scaler, outdir / "tf_scaler.pkl")

    return {
        "eval_mse": float(eval_loss),
        "eval_mae": float(eval_mae),
        "curve_path": str(curve_path),
        "model_path": str(model_path),
        "scaler_path": str(outdir / "tf_scaler.pkl"),
    }


def main():
    outdir = Path("./outputs").resolve()

    # 1) Carregar dados
    csv_path = Path("fabrica_energia.csv").resolve()
    print(f"Lendo dados de: {csv_path}")
    df = load_data(csv_path)

    # 1) Estatísticas básicas
    stats = compute_basic_stats(df)
    report_path = save_report(stats, outdir)
    print(f"Relatório salvo em: {report_path}")

    # 2) Gráficos
    scatter_path = plot_scatter(df, outdir)
    line_path = plot_line_consumo(df, outdir)
    print(f"Gráficos salvos:\n- {scatter_path}\n- {line_path}")

    # 3) Classificação (alto_consumo)
    df_cls = add_alto_consumo_label(df)
    clf, acc, cm_path = train_decision_tree(df_cls, outdir)
    print(f"Acurácia (Árvore de Decisão): {acc:.3f} — Matriz: {cm_path}")

    # 4) Predição (TensorFlow)
    tf_result = train_tensorflow_regressor(df, outdir)
    print("Predição com TensorFlow concluída.")
    print(json.dumps(tf_result, indent=2))

    # Exemplos de previsões rápidas
    example = pd.DataFrame({"horas_trabalhadas": [6, 8], "unidades_produzidas": [60, 90]})
    media = df["consumo_kwh"].mean()

    # Classe estimada pela árvore:
    pred_class = clf.predict(example.values)
    example["alto_consumo_previsto"] = pred_class
    example["rotulo_texto"] = np.where(example["alto_consumo_previsto"] == 1, "alto", "baixo")
    example["limiar_media_kwh"] = media
    example_path = outdir / "exemplos_classificacao.csv"
    example.to_csv(example_path, index=False, encoding="utf-8")
    print(f"Exemplos de classificação salvos em: {example_path}")


if __name__ == "__main__":
    main()
