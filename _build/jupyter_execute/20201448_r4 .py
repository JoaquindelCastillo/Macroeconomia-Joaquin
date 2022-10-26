#!/usr/bin/env python
# coding: utf-8

# # REPORTE N°4

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from sympy import *
import pandas as pd
#from causalgraphicalmodels import CausalGraphicalModel


# ## Reporte:

# El artículo escrito por Wund explica las medidas económicas tomadas por el FED en la crisis del 2008-2009, para lo cual busca explicarlo mediante una expansión del modelo tradicional de demanda y oferta agregada. Por ello la pregunta de investigación sería ¿cómo modelar la política monetaria no convencional seguida en la crisis internacional del 2008-2009?
# 
# Una de sus principales fortalezas fue presentar ejercicios para ver cómo se afectaban a las variables endógenas de su modelo. En estos no se centra en una descripción teórica o matemática, sino que también muestra gráficos sobre cómo cambiaban las curvas LM, IS y BB, lo cual permitía una mayor comprensión del modelo propuesto, además que le daba mayor validez académica al someterlo a diversas circunstancias y mostrando cómo funcionaba en esos contextos. Por otro lado, otra fortaleza fue la readaptación del modelo keynesiano, el cual tal como menciona el autor en muchas ocasiones suele ser catalogado como poco explicativo, justamente por su antigüedad, no obstante, Wund mediante su modelo de la política monetaria no convencional logra mostrar que este modelo sigue vigente. Sobre las debilidades identifico una: la contextualización. Si bien es cierto que el objetivo central del artículo era mostrar la funcionalidad del modelo de política monetaria no convencional, creo que hubiera sido óptimo mostrar una breve contextualización del suceso a estudiar que en este caso era la crisis del 2008-2009, es decir mostrar por qué y cómo sucedió para que el lector este familiarizado sobre la crisis a tratar. 
# 
# Ahora bien, el artículo avanza en la explicación de la pregunta de investigación, tras incluir al mercado de bonos a largo plazo al modelo IS-LM, lo cual permite entender mejor las cuestiones de políticas monetarias no convencionales usadas por el FED durante la crisis del 2008-2009. Por otro lado, este artículo no solo presenta el nuevo modelo IS-LM-BB sino que lo prueba mediante escenarios hipotéticos, por ejemplo mostrando los efectos de una menor tasa de interés a corto plazo o también viendo los efectos de un mayor gasto público. 
# 
# Ahora bien, uno de los primeros pasos para esta investigación sería ver más a fondo la crisis del 2008-2009 y el modelo propuesto en otros países cercanos que tuvieron una realidad distinta a Estados Unidos, por ejemplo, en el caso de Canadá , según autores como Ros (2012), tuvo varias diferencias en la exposición a la crisis en comparación con EEUU, por ejemplo, tuvo grados menores de exposición en los mercados crediticios de riesgo alto en comparación a los que prevalecían en Estados Unidos y de otros países europeos. Otra clara diferencia es que Canadá tenía una deuda pública baja proporcionalmente con su PBI, lo cual sumado a su situación fiscal relativamente holgada, le otorgaba un margen fiscal considerable para combatir el impacto de la recesión de la crisis del 2008-2009. 
# 
# Por otro lado, sería bueno poner a prueba dicho modelo en crisis actuales para ver su vigencia. Un claro ejemplo podría ser el COVID-19, la cual si bien fue una crisis sanitaria, también derivó en grandes problemas económicos que ocasionaron despidos masivos y con ello una crisis económica mundial. Esta crisis, según la ONU (2021), debe ser combatida mediante una política expansiva de gasto público muy grande, pues la crisis existente es de las más grandes de la historia. En este sentido una forma más de validar la importancia de modelos antiguos como el keynesiano para explicar problemas actuales, es ver a prueba dicho modelo con la Pandemia actual. 
#  
# Fuentes: 
# 
# Ros, Jaime. (2012). Junto al epicentro: análisis comparativo de las economías de Canadá y México durante la crisis de 2008-2009. Economía UNAM, 9(27), 22-44. Recuperado en 25 de septiembre de 2022, de http://www.scielo.org.mx/scielo.php?script=sci_arttext&pid=S1665-952X2012000300002&lng=es&tlng=es.
# 
# O.N.U (2021). Para impulsar la reactivación económica y mitigar los efectos negativos de la pandemia, es esencial que la región mantenga una política fiscal expansiva. Recuperado de 25 de septiembre de 2022, de https://www.cepal.org/es/comunicados/impulsar-la-reactivacion-economica-mitigar-efectos-negativos-la-pandemia-es-esencial-que
# 

# ## Solución parte código:
# 
#     - Sebastian Torres
#     - Joaquín Del Castillo
# 
# ### Parte 1:
# 
# #### 1. Encuentre las ecuaciones de Ingreso  y tasa de interes  de equilibrio.
# 
# Derivación del equilibrio del modelo IS-LM:
# 
# 
# 
# En primer lugar vemos la ecuación de la curva IS,
# $$ r = \frac{B_o}{h} - \frac{B_1}{h}Y $$
# 
# - Donde $ B_0 = C_o + I_o + G_o + X_o $ y $ B_1 = 1 - (b - m)(1 - t) $
# 
# 
# Y, por otro lado, la ecuación de la curva LM:
# 
# $$  r = -\frac{1}{j}\frac{Mo^s}{P_o} + \frac{k}{j}Y $$
# 
# Podemos igualar, sustituir o reducir ambas ecuaciones para encontrar el nivel de Ingresos equilibrio $(Y^e)$ y la tasa de interés de equilibrio $(r^e)$:
# 
# 
# 
# $$ -\frac{1}{j}\frac{Mo^s}{P_o} + \frac{k}{j}Y = \frac{B_o}{h} - \frac{B_1}{h}Y $$
# 
# - Ingreso de equilibrio:
# 
# $$ Y^e = \frac{j B_o}{k h + j B_1} + (\frac{h}{k h + j B_1})\frac{Ms_o}{P_o} $$
# 
# - Tasa de interés de equilibrio:
# 
# $$ r^e = \frac{kB_o}{kh + jB_1} - (\frac{B_1}{kh + jB_1})\frac{Ms_o}{P_o} $$
# 
# Estas dos ecuaciones representan el modelo IS-LM
# 

# #### 2. Grafique el equilibrio simultáneo en los mercados de bienes y de dinero.

# In[2]:


#--------------------------------------------------
    # Curva IS

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.4
m = 0.5
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo)/h - ( ( 1-(b-m)*(1-t) ) / h)*Y  
    return r_IS

r_is = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#--------------------------------------------------
    # Curva LM 

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 180            
P  = 25               

Y = np.arange(Y_size)

# Ecuación

def r_LM(k, j, Ms, P, Y):
    r_LM = - (1/j)*(Ms/P) + (k/j)*Y
    return r_LM

r_lm = r_LM( k, j, Ms, P, Y)


# In[3]:


# Gráfico del modelo IS-LM

# Dimensiones del gráfico
y_max = np.max(r_lm)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
# Curva IS
ax.plot(Y, r_is, label = "IS", color = "C1") #IS
# Curva LM
ax.plot(Y, r_lm, label="LM", color = "C0")  #LM

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Texto y figuras agregadas
# Graficar la linea horizontal - r
plt.axvline(x=51.5,  ymin= 0, ymax= 0.52, linestyle = ":", color = "black")
# Grafica la linea vertical - Y
plt.axhline(y=93, xmin= 0, xmax= 0.52, linestyle = ":", color = "black")

# Plotear los textos 
plt.text(49,100, '$E_0$', fontsize = 14, color = 'black')
plt.text(0,100, '$r_0$', fontsize = 12, color = 'black')
plt.text(53,-10, '$Y_0$', fontsize = 12, color = 'black')

# Título, ejes y leyenda
ax.set(title="Grafico simultaneo en los mercados de dinero y de bienes", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# ### Parte 2: Estática comparativa.

# #### 1) Analice los efectos sobre las variables endógenas Y, r de una disminución del gasto fiscal. El análisis debe ser intuitivo, matemático y gráfico.
#     - Intuitivo: 
# 
# $$ G↓ → DA↓ → DA<Y → Y↓ → DA=Y $$
# 
# $$ Y↓ → Md↓ → Md < Ms → r↓ → Md=Ms $$
# 
#     - Matemático: 

# In[4]:


# nombrar variables como símbolos
Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols('Co Io Go Xo h r b m t beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y = symbols('k j Ms P Y')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
Y_eq = (j*beta_0)/(k*h + j*beta_1) - ( beta_1 / (k*h + j*beta_1) )*(Ms/P)
r_eq = (k*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms/P)    


# In[5]:


df_Y_eq_Go = diff(Y_eq, Go)
print("El Diferencial del producto con respecto al diferencial del gasto autonomo = ", df_Y_eq_Go)  # este diferencial es positivo


# In[6]:


df_r_eq_Go = diff(r_eq, Go)
print("El Diferencial de la tasa de interes con respecto al diferencial del gasto autonomo = ", df_r_eq_Go)  # este diferencial es positivo


# Entonces: 
# 
# $$\frac{ΔY_e}{ΔG_0}=\frac{j}{kh+jB_1}ΔG_0<0 → Y (-)$$
# 
# $$\frac{ΔY_e}{ΔG_0}= (+)(-) < 0 → (-)$$
# 
# $$\frac{ΔY_e}{ΔG_0}=\frac{k}{kh+jB_1}ΔG_0 → 0$$
# 
# $$\frac{ΔY_e}{ΔG_0}= (+)(-) < 0 → (-)$$

#     - Gráfico: 

# In[7]:


#1--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.5
m = 0.4
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#2--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)


# In[8]:


#--------------------------------------------------
    # NUEVA curva IS: reducción Gasto de Gobienro (Go)
    
# Definir SOLO el parámetro cambiado
Go = 30

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_G = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[9]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS_(G_0)", color = "C1") #IS_orginal
ax.plot(Y, r_G, label = "IS_(G_1)", color = "C1", linestyle = 'dashed') #IS_modificada

ax.plot(Y, i, label="LM", color = "C0")  #LM_original

# Texto y figuras agregadas
plt.axvline(x=44.5,  ymin= 0, ymax= 0.45, linestyle = ":", color = "grey")
plt.axhline(y=78, xmin= 0, xmax= 0.45, linestyle = ":", color = "grey")

plt.axvline(x=53,  ymin= 0, ymax= 0.52, linestyle = ":", color = "grey")
plt.axhline(y=94, xmin= 0, xmax= 0.53, linestyle = ":", color = "grey")
plt.text(42,70, '$E_1$', fontsize = 14, color = 'black')

plt.text(50,100, '$E_0$', fontsize = 14, color = 'black')
plt.text(-1,100, '$r_0$', fontsize = 12, color = 'black')
plt.text(53,-40, '$Y_0$', fontsize = 12, color = 'black')
#plt.text(50,52, '$E_1$', fontsize = 14, color = '#3D59AB')
#plt.text(-1,72, '$r_1$', fontsize = 12, color = '#3D59AB')
plt.text(47,-40, '$Y_1$', fontsize = 12, color = '#3D59AB')

#plt.text(69, 115, '→', fontsize=15, color='grey')
plt.text(65, 60, '↙', fontsize=18, color='grey')

# Título, ejes y leyenda
ax.set(title="Politica Fiscal contractiva", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# #### 2) Analice los efectos sobre las variables endógenas Y, r de una disminución de la masa monetaria (ΔMs<0). El análisis debe ser intuitivo, matemático y gráfico.
#     - Intuitivo: 
#     
# $$Ms↓ → M°↓ → M°<Md → r↑$$
# 
# $$r↑ → I↓ → DA < Y → Y↓$$
# 
#     - Matemático: 

# In[10]:


Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols('Co Io Go Xo h r b m t beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y = symbols('k j Ms P Y')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
r_eq = (k*beta_0)/(k*h + j*beta_1) - ( beta_1 / (k*h + j*beta_1) )*(Ms/P)
Y_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms/P)


# In[11]:


df_r_eq_Ms = diff(r_eq, Ms)
print("El Diferencial de la tasa de interes con respecto al diferencial de la masa monetaria = ", df_r_eq_Ms)  # este diferencial es negativo


# In[12]:


df_Y_eq_Ms = diff(Y_eq, Ms)
print("El Diferencial del producto con respecto al diferencial de la masa monetaria = ", df_Y_eq_Ms)  # este diferencial es positivo


# Entonces: 
# 
# $${Δr}=\frac{k}{kh+jB_1}ΔM_0 → 0$$
# 
# $$Δr= (+) * (-) = (-) > 0$$
# 
# $${ΔY_e}=\frac{j}{kh+jB_1}ΔMo<0$$
# 
# $$YΔ= (+) * (-) = (-) < 0$$

#     - Gráfico: 

# In[13]:


#1--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.6
m = 0.5
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#2--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)


# In[14]:


# Definir SOLO el parámetro cambiado
Ms = 50

# Generar nueva curva LM con la variacion del Ms
def i_LM_Ms( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i_Ms = i_LM_Ms( k, j, Ms, P, Y)


# In[15]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS", color = "C1") #IS_orginal
ax.plot(Y, i, label="LM_(MS_0)", color = "C0")  #LM_original

ax.plot(Y, i_Ms, label="LM_(MS_1)", color = "C0", linestyle = 'dashed')  #LM_modificada

# Lineas de equilibrio_0 
plt.axvline(x=50,  ymin= 0, ymax= 0.52, linestyle = ":", color = "grey")
plt.axhline(y=93.5, xmin= 0, xmax= 0.53, linestyle = ":", color = "grey")

# Lineas de equilibrio_1 
plt.axvline(x=52.5,  ymin= 0, ymax= 0.51, linestyle = ":", color = "grey")
plt.axhline(y=98, xmin= 0, xmax= 0.5, linestyle = ":", color = "grey")
plt.text(62,87, '$E_1$', fontsize = 14, color = 'black')

#plt.axhline(y=68, xmin= 0, xmax= 0.52, linestyle = ":", color = "grey")

# Textos ploteados
plt.text(55,95, '$E_0$', fontsize = 14, color = 'black')
plt.text(-1,85, '$r_0$', fontsize = 12, color = 'black')
plt.text(55,20, '$Y_0$', fontsize = 12, color = 'black')
#plt.text(50,52, '$E_1$', fontsize = 12, color = 'black')
plt.text(-1,100, '$r_1$', fontsize = 12, color = 'black')
plt.text(46,20, '$Y_1$', fontsize = 12, color = 'black')

#plt.text(69, 11, '→', fontsize=15, color='grey')
plt.text(52, 98, '←', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title="Efecto de un descenso de la masa monetaria", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# #### 3) Analice los efectos sobre las variables endógenas Y, r de un incremento de la tasa de impuestos  (Δts>0). El análisis debe ser intuitivo, matemático y gráfico.
#     - Intuitivo: 
#     
# $$t↑ → DA↓ → DA<Y → Y↓ → DA=Y$$
# 
# $$Y↓ → Md↓ → Md < Ms → r↓ → Md=Ms$$
# 
#     - Matemático: 

# In[16]:


# nombrar variables como símbolos
Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols('Co Io Go Xo h r b m t beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y = symbols('k j Ms P Y')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
r_eq = (k*beta_0)/(k*h + j*beta_1) - ( beta_1 / (k*h + j*beta_1) )*(Ms/P)
Y_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms/P)


# In[17]:


df_Y_eq_t = diff(Y_eq, t)
print("El Diferencial del Producto con respecto al diferencial de la tasa de impuestos = ", df_Y_eq_t)  # este diferencial es negativo


# In[18]:


df_r_eq_t = diff(r_eq, t)
print("El Diferencial de la tasa de interes con respecto al diferencial de la tasa de impuestos = ", df_r_eq_t)  # este diferencial es positivo


# Entonces: 
# 
# $${ΔY_e}=\frac{j}{kh+ Δt} + \frac{h}{kh + Δt} Δt>0$$
# 
# $$YΔ= {-} + {-}  = (-) < 0$$
# 
# $${Δr}=\frac{k}{kh+ Δt} + \frac{Δt}{kh + Δt} Δt> 0$$
# 
# $$Δr= {-} + {-}  = (-) < 0$$
# 
#     - Gráfico:

# In[19]:


#1--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.4
m = 0.3
t = 0.1

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#2--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)


# In[20]:


#--------------------------------------------------
    # NUEVA curva IS
    
# Definir SOLO el parámetro cambiado
t = 0.9

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_G = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[21]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS_(G_0)", color = "C1") #IS_orginal
ax.plot(Y, r_G, label = "IS_(G_1)", color = "C1", linestyle = 'dashed') #IS_modificada

ax.plot(Y, i, label="LM", color = "C0")  #LM_original

# Texto y figuras agregadas
plt.axvline(x=54,  ymin= 0, ymax= 0.54, linestyle = ":", color = "grey")
plt.axhline(y=98, xmin= 0, xmax= 0.54, linestyle = ":", color = "grey")

plt.axvline(x=52,  ymin= 0, ymax= 0.52, linestyle = ":", color = "grey")
plt.axhline(y=94, xmin= 0, xmax= 0.52, linestyle = ":", color = "grey")
plt.text(50,85, '$E_1$', fontsize = 14, color = 'black')

plt.text(52,102, '$E_0$', fontsize = 14, color = 'black')
plt.text(-1,100, '$r_0$', fontsize = 12, color = 'black')
plt.text(55,10, '$Y_0$', fontsize = 12, color = 'black')
#plt.text(50,52, '$E_1$', fontsize = 14, color = 'black')
plt.text(-1,85, '$r_1$', fontsize = 12, color = 'black')
plt.text(48,10, '$Y_1$', fontsize = 12, color = 'black')

#plt.text(69, 115, '→', fontsize=15, color='grey')
plt.text(58, 88, '↙', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title="Incremento de la tasa impositiva", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# ## Puntos extra!

# ### Parte 1:
# 1. Encuentre las ecuaciones de Ingreso  y tasa de interes  de equilibrio.

# - Curva IS:
# 
# A partir de la nueva identidad Ingreso-Gasto: $ Y = C + I + G $
# 
# $$ Y = C_0 + bY^d + I_0 - hr + G_0$$
# 
# $$ Y = C_0 + I_0 + G_0 - hr + b(1-t)Y $$
# 
# $$ hr = C_0 + I_0 + G_0 + b(1-t)Y - Y $$
# 
# $$ hr = C_0 + I_0 + G_0 - Y(1- b(1-t)) $$
# 
# La ecuación de la curva IS es:
# 
# $$ r = \frac{C_0 + I_0 + G_0}{h} - \frac{1- b(1-t)}{h}Y $$
# 
# $$ r = \frac{B_0}{h} - \frac{B_1}{h}Y $$
# 
# Donde $B_0 = C_0 + I_0 + G_0 $ y $ B_1 = 1- b(1-t) $

# - Curva LM:
# 
# $$ \frac{M^s_0}{P_0} = kY - j(r + π^e) $$
# 

# $$ j(r + π^e) = kY - \frac{M^s_0}{P_0} $$
# 
# $$ r + π^e = - \frac{M^s_0}{jP_0} + \frac{kY}{j} $$
# 
# La ecuación de la curva LM es:
# 
# $$ r = - \frac{M^s_0}{jP_0} + \frac{k}{j}Y - π^e $$
# 

# - Equilibrio modelo IS-LM:
# 
# Para hallar $Y^e$:
# 

# $$ \frac{B_0}{h} - \frac{B_1}{h}Y = - \frac{M^s_0}{jP_0} + \frac{k}{j}Y - π^e $$
# 

# $$ \frac{B_0}{h} + \frac{M^s_0}{jP_0} + π^e = \frac{k}{j}Y + \frac{B_1}{h}Y $$

# $$ Y(\frac{k}{j} + \frac{B_1}{h}) = \frac{B_0}{h} + \frac{M^s_0}{jP_0} + π^e $$

# $$ Y(\frac{hk + jB_1}{jh}) = \frac{B_0}{h} + \frac{M^s_0}{jP_0} + π^e $$

# $$ Y^e = \frac{jB_0}{kh + jB_1} + \frac{M_0^s}{P_0} \frac{h}{kh + jB_1} + \frac{jh}{kh + jB_1} π^e $$

# Para hallar $r^e$:

# $$ r^e = - \frac{Ms_o}{P_o} (\frac{B_1}{kh + jB_1}) + \frac{kB_o}{kh + jB_1} - \frac{B_1}{kh + jB_1} π^e $$
# 

# #### 1.2 Grafique el equilibrio simultáneo en los mercados de bienes y de dinero.

# In[22]:


#--------------------------------------------------
    # Curva IS

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
h = 0.8
b = 0.5
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS_2(b, t, Co, Io, Go, h, Y):
    r_IS_2 = (Co + Io + Go - Y * (1-b*(1-t)))/h
    return r_IS_2

r_2 = r_IS_2(b, t, Co, Io, Go, h, Y)


#--------------------------------------------------
    # Curva LM 

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 20
π = 4

Y = np.arange(Y_size)

# Ecuación

def i_LM_2(k, j, Ms, P, Y, π):
    i_LM_2 = (-Ms/P)/j + k/j*Y - π
    return i_LM_2

i_2 = i_LM_2( k, j, Ms, P, Y, π)


# In[23]:


# Gráfico del modelo IS-LM

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r_2, label = "IS", color = "C1") #IS
ax.plot(Y, i_2, label="LM", color = "C0")  #LM

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Texto y figuras agregadas
plt.axvline(x=55,  ymin= 0, ymax= 0.54, linestyle = ":", color = "black")
plt.axhline(y=94, xmin= 0, xmax= 0.55, linestyle = ":", color = "black")
plt.text(53,102, '$E_0$', fontsize = 14, color = 'black')
plt.text(0,100, '$r_0$', fontsize = 12, color = 'black')
plt.text(56,-15, '$Y_0$', fontsize = 12, color = 'black')

# Título, ejes y leyenda
ax.set(title="Equilibrio modelo IS-LM", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# In[24]:


# nombrar variables como símbolos
Co, Io, Go, h, r, b, t, beta_0, beta_1  = symbols('Co, Io, Go, h, r, b, t, beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y, π = symbols('k j Ms P Y π')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go)
beta_1 = (1 - b*(1-t))

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
r_eq = -(Ms/P)*(beta_1/(k*h+j*beta_1)) + ((k*beta_0)/k*h+j*beta_1) - ((beta_1*π)/k*h+j*beta_1)
Y_eq = ((j*beta_0)/(k*h+j*beta_1)) + (Ms/P)*(h/(k*h+j*beta_1)) + (j*h*π/(k*h+j*beta_1))


# ### 2. Estática comparativa:

# #### 2.1. Analice los efectos sobre las variables endógenas Y, r de una disminución de los Precios $(∆P_0 < 0)$. El análisis debe ser intuitivo, matemático y gráfico.

# - Matemática:

# In[25]:


# nombrar variables como símbolos
Co, Io, Go, h, r, b, t, beta_0, beta_1  = symbols('Co, Io, Go, h, r, b, t, beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y, π = symbols('k j Ms P Y π')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go)
beta_1 = (1 - b*(1-t))

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
r_eq = -(Ms/P)*(beta_1/(k*h+j*beta_1)) + ((k*beta_0)/k*h+j*beta_1) - ((beta_1*π)/k*h+j*beta_1)
Y_eq = ((j*beta_0)/(k*h+j*beta_1)) + (Ms/P)*(h/(k*h+j*beta_1)) + (j*h*π/(k*h+j*beta_1))


# In[26]:


df_Y_eq_P = diff(Y_eq, P)
print("El Diferencial del Producto con respecto al diferencial del nivel de precios = ", df_Y_eq_P)


# ¿$∆Y$ sabiendo que $∆P < 0$?
# 
# $$ \frac{∆Y}{∆P} = (-) $$
# 
# $$ \frac{∆Y}{(-)} = (-) $$
# 
# $$ ∆Y > 0 $$

# In[27]:


df_r_eq_P = diff(r_eq, P)
print("El Diferencial de la tasa de interés con respecto al diferencial del nivel de precios = ", df_r_eq_P)


# ¿$∆r$ sabiendo que $∆P < 0$?
# 
# $$ \frac{∆r}{∆P} = (+) $$
# 
# $$ \frac{∆r}{(-)} = (+) $$
# 
# $$ ∆r < 0 $$

# - Intuición:
# 
# $$ P↓ → M^s↑ → M^s > M^d → r↓ $$
# 
# $$ r↓ → I↑ → DA↑ → DA > Y → Y↑ $$

# In[28]:


#--------------------------------------------------
    # Curva IS

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
h = 0.8
b = 0.5
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS_2(b, t, Co, Io, Go, h, Y):
    r_IS_2 = (Co + Io + Go - Y * (1-b*(1-t)))/h
    return r_IS_2

r_2 = r_IS_2(b, t, Co, Io, Go, h, Y)


#--------------------------------------------------
    # Curva LM 

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 20
π = 4

Y = np.arange(Y_size)

# Ecuación

def i_LM_2(k, j, Ms, P, Y, π):
    i_LM_2 = (-Ms/P)/j + k/j*Y - π
    return i_LM_2

i_2 = i_LM_2( k, j, Ms, P, Y, π)


#--------------------------------------------------
    # Nueva curva LM 
    
P = 5

# Ecuación

def i_LM_2(k, j, Ms, P, Y, π):
    i_LM_2 = (-Ms/P)/j + k/j*Y - π
    return i_LM_2

i_2_P = i_LM_2( k, j, Ms, P, Y, π)


# In[29]:


# Gráfico del modelo IS-LM

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r_2, label = "IS", color = "C1") #IS
ax.plot(Y, i_2, label="LM", color = "C0")  #LM
ax.plot(Y, i_2_P, label="LM", color = "C0", linestyle ='dashed')  #LM

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Texto y figuras agregadas
plt.axvline(x=55,  ymin= 0, ymax= 0.6, linestyle = ":", color = "black")
plt.axhline(y=94, xmin= 0, xmax= 0.55, linestyle = ":", color = "black")
plt.text(53,102, '$E_0$', fontsize = 14, color = 'black')
plt.text(0,100, '$r_0$', fontsize = 12, color = 'black')
plt.text(56,-45, '$Y_0$', fontsize = 12, color = 'black')

plt.axvline(x=64.5,  ymin= 0, ymax= 0.56, linestyle = ":", color = "black")
plt.axhline(y=85, xmin= 0, xmax= 0.64, linestyle = ":", color = "black")
plt.text(62,90, '$E_1$', fontsize = 14, color = 'C0')
plt.text(0,75, '$r_1$', fontsize = 12, color = 'C0')
plt.text(66,-45, '$Y_1$', fontsize = 12, color = 'C0')

# Título, ejes y leyenda
ax.set(title="Disminución del Precio", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# #### 2.2 Analice los efectos sobre las variables endógenas Y, r de una disminución de la inflación esperada $(∆π < 0)$. El análisis debe ser intuitivo, matemático y gráfico.

# - Matemática:

# In[30]:


# nombrar variables como símbolos
Co, Io, Go, h, r, b, t, beta_0, beta_1  = symbols('Co, Io, Go, h, r, b, t, beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y, π = symbols('k j Ms P Y π')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go)
beta_1 = (1 - b*(1-t))

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
r_eq = -(Ms/P)*(beta_1/(k*h+j*beta_1)) + ((k*beta_0)/k*h+j*beta_1) - ((beta_1*π)/k*h+j*beta_1)
Y_eq = ((j*beta_0)/(k*h+j*beta_1)) + (Ms/P)*(h/(k*h+j*beta_1)) + (j*h*π/(k*h+j*beta_1))


# In[31]:


df_Y_eq_π = diff(Y_eq, π)
print("El Diferencial del Producto con respecto al diferencial del nivel de inflación = ", df_Y_eq_π)


# ¿$∆Y$ sabiendo que $∆π < 0$?
# 
# $$ \frac{∆Y}{∆π} = (+) $$
# 
# $$ \frac{∆Y}{(-)} = (+) $$
# 
# $$ ∆Y < 0 $$

# In[32]:


df_r_eq_π = diff(r_eq, π)
print("El Diferencial de la tasa de interés con respecto al diferencial del nivel de inflación = ", df_r_eq_π)


# ¿$∆r$ sabiendo que $∆π < 0$?
# 
# $$ \frac{∆r}{∆π} = (-) $$
# 
# $$ \frac{∆r}{(-)} = (-) $$
# 
# $$ ∆r > 0 $$

# - Intuición:
# 
# $$ π↓ → r↑ $$
# 
# $$ r↑ → I↓ → DA↓ → DA < Y → Y↓ $$

# In[33]:


#--------------------------------------------------
    # Curva IS

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
h = 0.8
b = 0.5
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS_2(b, t, Co, Io, Go, h, Y):
    r_IS_2 = (Co + Io + Go - Y * (1-b*(1-t)))/h
    return r_IS_2

r_2 = r_IS_2(b, t, Co, Io, Go, h, Y)


#--------------------------------------------------
    # Curva LM 

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 20
π = 20

Y = np.arange(Y_size)

# Ecuación

def i_LM_2(k, j, Ms, P, Y, π):
    i_LM_2 = (-Ms/P)/j + k/j*Y - π
    return i_LM_2

i_2 = i_LM_2( k, j, Ms, P, Y, π)


#--------------------------------------------------
    # Nueva curva LM 
    
π = 2

# Ecuación

def i_LM_2(k, j, Ms, P, Y, π):
    i_LM_2 = (-Ms/P)/j + k/j*Y - π
    return i_LM_2

i_2_π = i_LM_2( k, j, Ms, P, Y, π)


# In[34]:


# Gráfico del modelo IS-LM

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r_2, label = "IS", color = "C1") #IS
ax.plot(Y, i_2, label="LM", color = "C0")  #LM
ax.plot(Y, i_2_π, label="LM", color = "C0", linestyle ='dashed')  #LM

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Texto y figuras agregadas
plt.axvline(x=54,  ymin= 0, ymax= 0.57, linestyle = ":", color = "black")
plt.axhline(y=95, xmin= 0, xmax= 0.54, linestyle = ":", color = "black")
plt.text(52,103, '$E_1$', fontsize = 14, color = 'C0')
plt.text(0,100, '$r_1$', fontsize = 12, color = 'C0')
plt.text(50,-35, '$Y_1$', fontsize = 12, color = 'C0')

plt.axvline(x=60,  ymin= 0, ymax= 0.55, linestyle = ":", color = "black")
plt.axhline(y=89, xmin= 0, xmax= 0.6, linestyle = ":", color = "black")
plt.text(58,95, '$E_0$', fontsize = 14, color = 'black')
plt.text(0,80, '$r_0$', fontsize = 12, color = 'black')
plt.text(56,-35, '$Y_0$', fontsize = 12, color = 'black')

# Título, ejes y leyenda
ax.set(title="Disminución de la inflación esperada", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()

