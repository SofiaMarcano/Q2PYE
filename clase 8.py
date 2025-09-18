# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 18:48:30 2025

@author: fjose

PRUEBAS ESTADÍSTICA
"""
#Importamos todas las librerías necesarias 

import numpy as np 

import pandas as pd 
#import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest # Prueba de z de proporcines 
from scipy.stats import wilcoxon
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests #para corrección bonferroni
from scipy.stats import ttest_ind #prueba t para dos muestras independientes 
from itertools import  combinations #combinaciones para prueba t de comparaciones multiples
from scipy.stats import ttest_rel #Prueba t para muestras pareadas o no independientes 
from scipy.stats import mannwhitneyu #Prueba U de Mann-Whitney 
#%% Importar datos 
# por ejemplo:
df = pd.read_excel("visceral_fat.xlsx")

#df1 = df[df["age (years)"]<=28] # mujeres con hasta 28 años de edad
#df2 = df[df["age (years)"]>28]# mujeres con más de 28 años de edad

df1 = df[df["ethnicity"]==1]#
df2 = df[df["ethnicity"]==0]#
#si la variable tiene NaN las pruebas no arrojaran un resultado, se deben imputar esos datos 
# por ejemplo con la media así

#en al bmi hay un NAN entonces 
df['bmi pregestational (kg/m)'].fillna(df['bmi pregestational (kg/m)'].mean(),inplace=True)
#%% Funciones 
def t_test_one(data,mu,variable): #Prueba T para una muestra
    """
    data: arreglo de datos a comparar
    mu: media poblacional o valor de referencia 
    variable: string con el nombre de la variable que se está comparando
    """
    print(f"Prueba T para una sola muestra para {variable}")
    t_stat, p_value = stats.ttest_1samp(data, mu)
    print(f"Estadístico = {t_stat:.4f}, valor_p = {p_value:.4f}")

# Pruebas de normalidad
def test_normalityKS(data, variable): # Pruaba de Normalidad Kolmogorov-Smirnof 
    """
    data: arreglo de datos a evaluar la normalidad
    variable: string con el nombre de la variable 
    """  
    print(f"\n Análisis de normalidad por Kolmogorov-Smirnov para '{variable}'")

    # Kolmogorov-Smirnov (KS) test
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    print(f" Estadístico = {ks_stat:.4f}, p-valor = {ks_p:.4f}")

def test_normalitySW(data, variable): # Prueba de Normalizas Shapiro-Wilks 
    """
    data: arreglo de datos a evaluar la normalidad
    variable: string con el nombre de la variable 
    """
    print(f"\n Análisis de normalidad por Shapiro-Wilk para '{variable}'")
    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(data)
    print(f"Estadístico = {shapiro_stat:.4f}, p-valor = {shapiro_p:.4f}")
    
def proportions(exitos,n,proportion,variable,direccion): # Prueba de proporciones
    """
    exitos: número de éxitos
    n: número de ensayos 
    proportion: proporcion a corroborar 
    Variable: string con el nombre de  la variable 
    dirección: string para la dirección de la prueba: "two-sided", "larger", "smaller"
    """

    print(f"Prueba de proporciones para {variable}")
    stat, p_value = proportions_ztest(count=exitos, nobs=n, value=proportion, alternative='two-sided')
    print(f"Estadístico = {stat:.4f}, Valor_p = {p_value:.4f}")

def binomial(exitos,k,prop,direccion):
    """
    exitos: número de éxitos
    k: número de ensayos 
    prop: proporcion a corroborar  
    dirección: string para la dirección de la prueba: "two-sided", "less", "greater"
    """
    
    result = binomtest(exitos, n=k, p=prop, alternative=direccion)
    print("Prueba Binomial")
    print(f"P-valor: {result.pvalue:.4f}")

    
def sgn_test (before,after,direccion): #Prueba de Signos 
    """
    before: variable antes del tratamiento
    after: variable después del tratamiento, o mediana de comparación
    direccion: dirección de la prueba: "two-sided", "less", "greater"
    """
    #Calcular diferencias
    diff = before - after
    # Contar signos
    signos_positivos = np.sum(diff > 0)
    signos_negativos = np.sum(diff < 0)
    
    # Prueba binomial
    result = binomtest(min(signos_positivos, signos_negativos), n=signos_positivos + signos_negativos, p=0.5, alternative=direccion)
    print(f"Signos positivos: {signos_positivos}, Signos negativos: {signos_negativos}")
    print(f"P-valor: {result.pvalue:.4f}")

def wilcoxon_test (data,mediana=0,y=None):# prueba Wilcoxon 
    """
    data: datos de la variable a comparar, puede ser la primera (antes) si se hara para dos
    mediana=valor de la mediana estándar o de referencia, si la prueba es de dos muestras no se modifica mediana
    y=los valores de la segunda variable a comparar (después), si la muestra es de solo una muestra, entonces None
    """
    # Prueba de Wilcoxon (hipótesis:
    stat, p_value = wilcoxon(data - mediana,y)
    
    print(f"Estadístico de Wilcoxon: {stat:.4f}, p-valor: {p_value:.4f}")
    
def test_homogeneityL(var1, var2, name1, name2): # Prueba de levene
    """
    var1 y var2: variables a las que se corroborará homocedasticidad 
    name1 y name2: strings con el nnombre de las variables
    """
    print(f"\n Análisis de homocedasticidad entre '{name1}' y '{name2}'")

    # Prueba de Levene (no asume normalidad)
    levene_stat, levene_p = stats.levene(var1, var2)
    print(f"Levene test: Estadístico = {levene_stat:.4f}, p-valor = {levene_p:.4f}")

def test_homogeneityB(var1, var2, name1, name2): # Prueba de Barttlet 
    """
    var1 y var2: variables a las que se corroborará homocedasticidad 
    name1 y name2: strings con el nnombre de las variables
    """
    print(f"\n Análisis de homocedasticidad entre '{name1}' y '{name2}'")
    # Prueba de Bartlett (requiere normalidad)
    bartlett_stat, bartlett_p = stats.bartlett(var1, var2)
    print(f"Bartlett test: Estadístico = {bartlett_stat:.4f}, p-valor = {bartlett_p:.4f}")
def t_test_two_sample(datos1,datos2):#prubea T para dos muestras
    """
    datos1: datos de la primera muestra 
    datos2: datos de la segunda muestra
    """
    
    t_stat, p_value = ttest_ind(datos1, datos2)
    print(f"Prueba t para dos muestras, valor p: {p_value:.4f}")
    
def t_test_multiple (df,tto,variable): #Prueba T de comparaciones multiples
    """
    df: dataframe, en cada columna debe estar cada variable a comparar 
    tto: string con el nombre de la columna del tratamiento
    variable: string con el nombre de la variable
    
    """    

    valores_unicos = df[tto].unique()#determinar cuales son los tratamientos posibles

    # Lista de combinaciones de pares de tratamientos
    comparaciones = list(combinations(valores_unicos, 2))
    p_values = []#lista para almacener los valores P de las comparaciones
    
    for g1, g2 in comparaciones:#extraer los datos a comparar en cada iteracion
        # Extraer datos de cada grupo
        datos1 = df[df[tto] == g1][variable]
        datos2 = df[df[tto] == g2][variable]
        
        # Prueba t para muestras independientes para cada par
        t_stat, p_value = ttest_ind(datos1, datos2)
        p_values.append(p_value)
        print(f"Comparación {g1} vs {g2}: t={t_stat:.4f}, p={p_value:.4f}")
    
    # Aplicar corrección de Bonferroni
    p_corrected = multipletests(p_values, method='bonferroni')[1]
    
    print("\nP-valores corregidos con Bonferroni:")
    for i, (g1, g2) in enumerate(comparaciones):
        print(f"{g1} vs {g2}: p-corrected = {p_corrected[i]:.4f}")

def t_test_paired (before,after):#prueba T para datos pareados
    """
    before: variable antes 
    after: variable después
    """
    
    t_stat, p_value = ttest_rel(before, after)

    # Mostrar resultados
    
    print(f"Prueba t para muestras pareadas, P-valor: {p_value:.4f}")

def U_Mann_W (datos1,datos2):# pruba U Mann Withney
    """
    datos1: primer set de datos a comparar
    datos2: segundo set de datos a comparar
    """
    # Prueba de Mann-Whitney
    stat, p_value = mannwhitneyu(datos1, datos2, alternative='two-sided')
    # Mostrar resultados
    print(f"Estadístico U: {stat}")
    print(f"P-valor: {p_value:.4f}")


#%% 


#%% ejemplo para t de comparaciones multiples 
grupo_A = [120, 118, 115, 122, 121]
grupo_B = [130, 135, 128, 132, 134]
grupo_C = [125, 127, 126, 124, 123]

# Crear un DataFrame
df = pd.DataFrame({
    'Presión': grupo_A + grupo_B + grupo_C,
    'Tratamiento': ['A']*len(grupo_A) + ['B']*len(grupo_B) + ['C']*len(grupo_C)
})
#%%
t_test_multiple(df,"Tratamiento", "Presión")
