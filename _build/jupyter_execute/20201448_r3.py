#!/usr/bin/env python
# coding: utf-8

# # REPORTE N°3

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
import pandas as pd
import numpy as np
import random
import math
import sklearn
import scipy as sp
import networkx
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from causalgraphicalmodels import CausalGraphicalModel


# ### Integrantes: Joaquin del Castillo, Sebastian Torres

# ### Pregunta 1: 

# #### 1.1) 
# 
# Son cuatro los instrumentos de la política monetaria:
# 
# 
# 1)Operación de Mercado Abierto: El BCR realiza estas operaciones comprando y vendiendo activos financieros o bonos a los bancos  comerciales. En el caso de una política monetaria expansiva se mediante la compra de bonos, con la cual el BCR inyecta soles en la economía. Mientras que en la política monetaria contractiva se da con la venta de bonos al mercado (bancos comerciales). Con esta operación, el Banco Central retira dinero (soles) de la economía. 
# 
# 
# 
# 2)El coeficiente legal de encaje como instrumento de política es el segundo instrumento. En el caso de una política moentaria expansiva el BCR puede aumentar la cantidad de dinero que poseen los bancos para generar préstamos mediante la reducción de la tasa de encaje, ello genera que los bancos comerciales tengan menos dinero en reservas en el balance del BCR. Ahora bien, en la política monetaria contractiva el BCR aumenta el coneficiente de encaje, los bancos comerciales tienen una mayor proporción de depósitos en reservas y ello reduce el multiplicador bancario y origina una disminución en la oferta monetaria. 
# 
# 
# 3)La tasa de interés como instrumento de política:A partir del 1990 la tasa de interés se transformó en instrumento de política y la oferta monetaria pasó a ser una variable endógena. La tasa de política es la tasa de interés de referencia de la política monetaria. En el caso de la política monetaria expansiva el BCR reduce la tasa de interés de referencia, con lo cual aumenta el dinero prestado a los bancos comerciales y ello incrementa la base monetaria y de esta forma la oferta monetaria. Mientras que en el caso de la política monetaria contractiva el BCR aumenta la tasa de interes de referencia, con lo cual reduce el dinero prestado a los bancos comerciales y ello reduce la base monetaria y de esta forma la oferta monetaria. 
# 
# 
# 4)Función de Oferta Real de Dinero: El autor supone que la oferta nominal de dinero (𝑀𝑠) es una variable exógena e instrumento de política monetaria. En términos reales o a precios constantes, la oferta nominal se divide entre en nivel general de precios de la economía, (𝑃). Se supone que el nivel de precios estan dados a corto plazo. 
# 

# #### 1.2) 
# 
# $${M^s_0}$$
# 
# Se puede ver, en primer lugar, a la masa monetaria, la cual es el dinero que crea el BCR y se encuentra en circulación en la economía. 
# 
# $$P$$
# 
# Se puede ver, en segundo lugar, al precio, el cual indica el valor real de cada billete o divisa. 
# 
# $$M^s= \frac{M^s_0}{P}$$
# 
# Finalmente, se puede ver que la división entre el número de billetes que transitan en la economía con el precio que valen, se obtiene a la oferta real de dinero. 

# #### 1.3)
# 
# 1)El motivo Transacción: Dada su función de medio de intercambio, el dinero se demanda para realizar transacciones. La magnitud de las transacciones de bienes y servicios está en relación directa con el ingreso o producto de la economía. Por ende, la demanda de dinero depende directamente del Ingreso (Y). 
# 
# 2)El motivo de Precaución:  En este caso, el dinero se demanda para pagar las deudas. Como los que se endeudan deben tener capacidad de pago, y esta depende de sus ingresos, a nivel macroeconómico la demanda de dinero por el motivo precaución también depende positivamente del ingreso (Y). 

# Entonces, en un primer momento, la demanda de dinero por motivos de transacción y precaución será: 
# 
# $$ L_1 = kY $$ 

# 3) El motivo Especulativo: El activo financiero dinero compite con el activo financiero no monetario bono en la función de reserva de valor. Se preferirá mantener liquidez en forma de dinero y no en forma de bonos cuando la tasa de interés se reduce y lo contrario si aumenta. Por ende, esta demanda dependerá inversamente de la tasa de interés de los bonos. 

# Entonces, en un segundo momento, la demanda de dinero por motivo de especulació será:

# $$ L_2 = -ji $$

# Por tanto, si asumimos que ambos tipos de demanda están en términos reales, la función de demanda real de dinero será: 

# $$ L = L_1 + L_2 $$
# $$ L = kY - ji $$
# 
# - Donde: 
#     - $ k $ = indica la sensibilidad de la demanda de dinero ante variaciones del ingreso (“Y”).
#     - $ j $ = indica la sensibilidad de la demanda de dinero ante las variaciones de la tasa de interés nominal de los bonos ("i"). 

# #### 1.4) 
# 
# El equilibrio en el Mercado de Dinero se deriva del equilibrio entre la Oferta de Dinero $(M^s)$ y Demanda de Dinero $(M^d)$:
# 
# $$ M^s = M^d $$
# 
# $$ \frac{M^s}{P} = kY - ji $$

# Dado que en el Corto Plazo (CP) el nivel de precios es fijo y exógeno, podemos asumir que la inflación esperada será 0. Por ende, no habría una gran diferencia entre la tasa de interés nominal $(i)$ y la real $(r)$.
# 
# Entonces la ecuación del equilibrio en el mercado monetario es:
# 
# $$ \frac{M^s}{P} = kY - jr $$
# 

# #### 1.5) 

# In[2]:


# Parameters
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 35
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_0 = MD(k, j, P, r, Y)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS


# In[3]:


# Equilibrio en el mercado de dinero

# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo t el nombre de las coordenadas
ax1.set(title="Money Market Equilibrium", xlabel=r'M^s / P', ylabel=r'r')

# Ploteamos la demanda de dinero
ax1.plot(MD_0, label= '$L_0$', color = '#CD5C5C')

# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=7.5, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")

# Agregamos texto
ax1.text(0, 7.8, "$r_0$", fontsize = 12, color = 'black')
ax1.text(52, 0, "$(M_o^s) / P_0$", fontsize = 12, color = 'black')
ax1.text(85, 1, "$L(Y_0)$", fontsize = 12, color = 'black')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()

plt.show()


# ## Pregunta 2: 

# #### 2.1) ∆ Y < 0

# $ Explicación: $
# 
# Cuando disminuye el ingreso “ Y ”, la demanda de dinero también disminuye. Si el mercado estaba en equilibrio, esta disminución del ingreso genera un exceso de oferta en el mercado.  El equilibrio debe restablecerse con una disminución de la tasa de interés. La curva de demanda de dinero se desplaza hacia abajo.
# 
# $ Graficando: $

# In[4]:


# Parameters
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 50
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_0 = MD(k, j, P, r, Y)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS


# In[5]:


# Parameters con cambio en el nivel del producto
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y_1 = 35
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_1 = MD(k, j, P, r, Y_1)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS

# Equilibrio en el mercado de dinero

# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo t el nombre de las coordenadas
ax1.set(title="Money Market Equilibrium", xlabel=r'M^s / P', ylabel=r'r')

# Ploteamos la demanda de dinero
ax1.plot(MD_0, label= '$L_0$', color = '#CD5C5C')


# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=7.5, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")

# Agregamos texto
ax1.text(0, 7.8, "$r_1$", fontsize = 12, color = 'black')
ax1.text(50, 0, "$(Ms/P)_0$", fontsize = 12, color = 'black')
ax1.text(50, 7.5, "$E_1$", fontsize = 12, color = 'black')

# Nuevas curvas a partir del cambio en el nivel del producto
ax1.plot(MD_1, label= '$L_1$', color = '#4287f5')
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")
ax1.axhline(y=15, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")
ax1.text(0, 15.4, "$r_0$", fontsize = 12, color = 'black')
ax1.text(50, 15.3, "$E_0$", fontsize = 12, color = 'black')


plt.text(42, 12, '∆Y', fontsize=12, color='black')
plt.text(45, 11.9, '↓', fontsize=15, color='grey')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()

plt.show()


# #### 2.2) ∆ k < 0; si la elasticidad del ingreso disminuye:

# $ Explicación: $
# 
# Cuando disminuye la sensibilidad de la demanda de dinero ante variaciones del ingreso, la demanda de dinero también disminuye. Si el mercado estaba en equilibrio, esta disminución de la elasticidad del ingreso genera un exceso de oferta en el mercado (dado que a la economía le va mal, vamos a demandar menos dinero y, por ende, se reduce el nivel de consumo).  El equilibrio debe restablecerse con una disminución de la tasa de interés. La curva de demanda de dinero se desplaza hacia abajo.
# 
# $ Graficando: $

# In[6]:


# Parameters
r_size = 100

k = 0.8
j = 0.2                
P  = 10 
Y = 35
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_0 = MD(k, j, P, r, Y)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS


# In[7]:


# Parameters con cambio en k
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 35
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_1 = MD(k, j, P, r, Y_1)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS

# Equilibrio en el mercado de dinero

# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo t el nombre de las coordenadas
ax1.set(title="Money Market Equilibrium", xlabel=r'M^s / P', ylabel=r'r')

# Ploteamos la demanda de dinero
ax1.plot(MD_0, label= '$L_0$', color = '#CD5C5C')


# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=7.5, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")

# Agregamos texto
ax1.text(0, 7.8, "$r_1$", fontsize = 12, color = 'black')
ax1.text(50, 0, "$(Ms/P)_0$", fontsize = 12, color = 'black')
ax1.text(50, 7.5, "$E_1$", fontsize = 12, color = 'black')

# Nuevas curvas a partir del cambio en el nivel del producto
ax1.plot(MD_1, label= '$L_1$', color = '#4287f5')
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")
ax1.axhline(y=18, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")
ax1.text(0, 18.4, "$r_0$", fontsize = 12, color = 'black')
ax1.text(50, 18, "$E_0$", fontsize = 12, color = 'black')

# Texto agregado
plt.text(42, 12, '∆k', fontsize=12, color='black')
plt.text(45, 11.9, '↓', fontsize=15, color='grey')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())


ax1.legend()

plt.show()


# #### 2.3) ∆ Ms < 0; cantidad de dinero se disminuye: 

# $ Explicación: $
# 
# La disminución de la cantidad de dinero origina una disminución de la oferta real de dinero. Por ende, la oferta real de dinero será menor que la demanda real de dinero. El exceso de demanda que se produce en el mercado debe dar lugar a un aumento de la tasa de interés para que aumente la oferta y se restablezca el equilibrio en el mercado. En consecuencia, al disminuir la cantidad de dinero disminuye de 𝑀0𝑠 a 𝑀1𝑠, la recta de la oferta real de dinero se desplaza hacia la izquierda.
# 
# $ Graficando: $

# In[8]:


# Parameters con cambio en el nivel del producto
r_size = 100

k = 0.5
j = 0.2                
P_1 = 20 
Y = 35
MS_0 = 550

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_1 = MD(k, j, P_1, r, Y)
# Necesitamos crear la oferta de dinero.
MS_1 = MS_0 / P_1
MS


# In[9]:


# Equilibrio en el mercado de dinero

# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo t el nombre de las coordenadas
ax1.set(title="Money Market Equilibrium", xlabel=r'M^s / P', ylabel=r'r')

# Ploteamos la demanda de dinero
ax1.plot(MD_0, label= '$L_0$', color = '#CD5C5C')
#ax1.plot(MD_1, label= '$L_0$', color = '#CD5C5C')


# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=18.1, xmin= 0, xmax= 0.50, linestyle = ":", color = "black")

# Agregamos texto
ax1.text(0, 19, "$r_0$", fontsize = 12, color = 'black')
ax1.text(52, 8, "$(Ms/P)_0$", fontsize = 12, color = 'black')
ax1.text(51, 18.5, "$E_0$", fontsize = 12, color = 'black')


# Nuevas curvas a partir del cambio en el nivel del producto
#ax1.plot(MD_1, label= '$L_1$', color = '#4287f5')
ax1.axvline(x = MS_1,  ymin= 0, ymax= 1, color = "grey")
ax1.axhline(y=22.5, xmin= 0, xmax= 0.30, linestyle = ":", color = "black")
ax1.text(0, 23, "$r_1$", fontsize = 12, color = 'black')
ax1.text(29, 8, "$(Ms/P)_1$", fontsize = 12, color = 'black')
ax1.text(28.5, 22.5, "$E_1$", fontsize = 12, color = 'black')

plt.text(32, 19.5, '∆Ms', fontsize=12, color='black')
plt.text(32.4, 18.8, '←', fontsize=15, color='grey')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()

plt.show()


# ## 3. Curva LM: 

# #### 3.1)

# In[10]:


#1: 

#1----------------------Equilibrio mercado monetario

    # Parameters
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 35

r = np.arange(r_size)


    # Ecuación
def Ms_MD(k, j, P, r, Y):
    Ms_MD = P*(k*Y - j*r)
    return Ms_MD

Ms_MD = Ms_MD(k, j, P, r, Y)


    # Nuevos valores de Y
Y1 = 45

def Ms_MD_Y1(k, j, P, r, Y1):
    Ms_MD = P*(k*Y1 - j*r)
    return Ms_MD

Ms_Y1 = Ms_MD_Y1(k, j, P, r, Y1)


Y2 = 25

def Ms_MD_Y2(k, j, P, r, Y2):
    Ms_MD = P*(k*Y2 - j*r)
    return Ms_MD

Ms_Y2 = Ms_MD_Y2(k, j, P, r, Y2)

#2----------------------Curva LM

    # Parameters
Y_size = 100

k = 0.5
j = 0.2                
P  = 10               
Ms = 30            

Y = np.arange(Y_size)


# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)


# In[11]:


# Gráfico de la derivación de la curva LM a partir del equilibrio en el mercado monetario

    # Dos gráficos en un solo cuadro
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20, 8)) 


#---------------------------------
    # Gráfico 1: Equilibrio en el mercado de dinero
    
ax1.set(title="Money Market Equilibrium", xlabel=r'M^s / P', ylabel=r'r')
ax1.plot(Y, Ms_MD, label= '$L_0$', color = '#CD5C5C')
ax1.plot(Y, Ms_Y1, label= '$L_1$', color = '#CD5C5C')
ax1.plot(Y, Ms_Y2, label= '$L_2$', color = '#CD5C5C')
ax1.axvline(x = 45,  ymin= 0, ymax= 1, color = "grey")

ax1.axhline(y=35, xmin= 0, xmax= 1, linestyle = ":", color = "black")
ax1.axhline(y=135, xmin= 0, xmax= 1, linestyle = ":", color = "black")
ax1.axhline(y=85, xmin= 0, xmax= 1, linestyle = ":", color = "black")

ax1.text(47, 139, "C", fontsize = 12, color = 'black')
ax1.text(47, 89, "B", fontsize = 12, color = 'black')
ax1.text(47, 39, "A", fontsize = 12, color = 'black')

ax1.text(0, 139, "$r_2$", fontsize = 12, color = 'black')
ax1.text(0, 89, "$r_1$", fontsize = 12, color = 'black')
ax1.text(0, 39, "$r_0$", fontsize = 12, color = 'black')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()
 

#---------------------------------
    # Gráfico 2: Curva LM
    
ax2.set(title="LM SCHEDULE", xlabel=r'Y', ylabel=r'r')
ax2.plot(Y, i, label="LM", color = '#3D59AB')

ax2.axhline(y=160, xmin= 0, xmax= 0.69, linestyle = ":", color = "black")
ax2.axhline(y=118, xmin= 0, xmax= 0.53, linestyle = ":", color = "black")
ax2.axhline(y=76, xmin= 0, xmax= 0.38, linestyle = ":", color = "black")

ax2.text(67, 164, "C", fontsize = 12, color = 'black')
ax2.text(51, 122, "B", fontsize = 12, color = 'black')
ax2.text(35, 80, "A", fontsize = 12, color = 'black')

ax2.text(0, 164, "$r_2$", fontsize = 12, color = 'black')
ax2.text(0, 122, "$r_1$", fontsize = 12, color = 'black')
ax2.text(0, 80, "$r_0$", fontsize = 12, color = 'black')

ax2.text(72.5, -14, "$Y_2$", fontsize = 12, color = 'black')
ax2.text(56, -14, "$Y_1$", fontsize = 12, color = 'black')
ax2.text(39, -14, "$Y_0$", fontsize = 12, color = 'black')

ax2.axvline(x=70,  ymin= 0, ymax= 0.69, linestyle = ":", color = "black")
ax2.axvline(x=53,  ymin= 0, ymax= 0.53, linestyle = ":", color = "black")
ax2.axvline(x=36,  ymin= 0, ymax= 0.38, linestyle = ":", color = "black")

ax2.yaxis.set_major_locator(plt.NullLocator())   
ax2.xaxis.set_major_locator(plt.NullLocator())

ax2.legend()

plt.show()


# ####  3.1.2)

# Analitico: 
# 
# $$M^s=M^d$$
# 
# A continuación se muestra el equilibrio entre la oferta real de dinero y la demanda monetaria. 
# 
# $$\frac{M^s_0}{P_0}=(kY-ji)$$
# 
# Ahora descompusimos cada una de sus variables
# 
# $$\frac{M^s_0}{P_0}=(kY-jr)$$
# 
# A continuación cambiamos i por r, pues en nuestra ecuación valen lo mismo
# 
# $$\frac{kY}{j}-\frac{M^s_0}{P_0}{j}=r$$
# 
# A continuación hacemos operaciones aritmeticas y nos queda la curva lm en donde r es la variable endogena. 
# 
# $$r= \frac{-1}{j}\frac{M^s_0}{P_0}+ \frac{kY}{j} $$
# 
# 

# #### 3.2) ¿Cuál es el efecto de una disminución en la Masa Monetaria?
# 
# Explica usando la intuición y gráficos.

# $$ ↓M^s_0 → ↓\frac{M^s_0}{P_0} → M^s < M^d → r↑ $$

# La masa monetaria cae, con lo cual la oferta monetaria cae y ello genera un desequilibrio. Para solucionarlo debe subir la tasa de interés. 

# In[12]:


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

#--------------------------------------------------
    # NUEVA curva LM

# Definir SOLO el parámetro cambiado
Ms = 20

# Generar la ecuación con el nuevo parámetro
def i_LM_Ms( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i_Ms = i_LM_Ms( k, j, Ms, P, Y)


# In[13]:


# Dimensiones del gráfico
y_max = np.max(i)
v = [0, Y_size, 0, y_max]   
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, i, label="LM", color = 'black')
ax.plot(Y, i_Ms, label="LM_Ms", color = '#3D59AB', linestyle = 'dashed')

# Texto agregado
plt.text(41, 56, '∆$M^s$', fontsize=12, color='black')
plt.text(36, 66, '←', fontsize=15, color='black')

# Título y leyenda
ax.set(title = "Descenso en la Masa Monetaria $(M^s)$", xlabel=r'Y', ylabel=r'r')
ax.legend()


plt.show()


# #### 3.3)¿Cuál es el efecto de un aumento en k ? Explica usando intuición y gráficos.

# In[14]:


#--------------------------------------------------
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

#--------------------------------------------------
    # NUEVA curva LM

# Definir SOLO el parámetro cambiado
k = 4

# Generar la ecuación con el nuevo parámetro
def i_LM_Ms( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i_Ms = i_LM_Ms( k, j, Ms, P, Y)


# In[15]:


# Dimensiones del gráfico
y_max = np.max(i)
v = [0, Y_size, 0, y_max]   
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, i, label="LM", color = 'black')
ax.plot(Y, i_Ms, label="LM_Ms", color = '#3D59AB', linestyle = 'dashed')

# Texto agregado
plt.text(47, 140, '∆$k$', fontsize=12, color='black')
plt.text(47, 120, '←', fontsize=15, color='black')

# Título y leyenda
ax.set(title = "Aumento en k $(k)$", xlabel=r'Y', ylabel=r'r')
ax.legend()


plt.show()


# Intuición: 
#     
# $$ ↑ k → ↑kY → M^d > M^s → ↑ r $$

# Cuando sube k sube la demanda monetaria. Ello quiere decir que la curva lm se encuentra en desequilibrio, para lo cual es necesario que la tasa de interés suba.

# ## Reporte: 

# La pregunta de investigación del texto “La macroeconomia de la cuarentena: Un modelo de dos sectores sería: ¿Cómo explicar los efectos macroeconómicos sufridos en la económia peruana en base a la creación de un modelo con dos sectores (uno afectado de manera directa y el otro de manera indirecta)?
# 
# Ahora bien, las fortalezas que encuentro son, en primer lugar, la claridad y orden con la cual plantea su modelo macroeconomico, lo cual permite que este paper pueda ser entendido por la mayoria de lectores y no solo por economistas. Ello es importante, porque este artículo tiene como principal objetivo explicar los efectos macroeconómicos de la cuarenta, un tema que afectó a todos los peruanos, lo cual hace que sea un tema de interés público. 
# 
# En segundo lugar, el orden seguido en el artículo, pues desde la introducción nos comentan el orden que se va a seguir en el paper y en las conclusiones, finalmente, termina por responder a los objetivos planteados por los autores. Ello genera que la comprensión del artículo sea mucho más fácil. 
# 
# Las debilidades que encuentro son dos. En primer lugar, la presentación de las fórmulas matemáticas. Si bien como se comentó, el texto puede ser entendido en su mayoría por diferentes públicos, ello no sucede con las fórmulas matemáticas, las cuales en mi opinión necesitaban una mayor explicación para que pueda ser más entendible. En segundo lugar, que solo el modelo se centró en dos sectores, lo cual por ejemplo, generó que no se enfoquen mucho en sectores como el minero, el cual según el IPE paso de aportar 1.8% en el PBI (antes de pandemia) a solo 0.3% durante la pandemia. 
# 
# En segundo lugar, que la estática comparativa sea más realista, pues si bien se vio la respuesta de política pública en el ambito sanitario (la cuarentena) falto verlo en el ámbito económico, cuestión que los mismos autores señalan en las conclusiones. 
# 
# Su principal contribución fue probar efectivamente mediante la creación de su modelo los efectos de la cuarentena tanto en la producción como también en los precios tanto en el sector 1, el afectado directamente, y el sector 2, el afectado de manera indirecta por la caida de la demanda en el sector 1. Para lo cual crea dos ejercicios y en base a ello saca conclusiones sobre los efectos de la cuarentena: 
# 
# Sobre el primer ejercicio se ve que hay un descenso del producto potencial, acompañado simultaneamente por una caída del consumo autonomo permanente en el sector 1, en otras palabras sus niveles se encuentran en un nuevo equilibrio estacionario. En este escenario luego de los descensos de producción y precios en los dos sectores, la economía se recupera progresivamente en consecuencia de la caída de los precios esperados, que bajan los precios y elevan la riqueza real, el consumo, la demanda y producción en los dos sectores. 
# 
# En el segundo ejercicio, la caída del PBI potencial como también del consumo autónomo son transitorias, pues se restablecen cada 4 periodos. El comportamiento en este segundo ejercicio tiene más relación con la recuperación vigoroso, al punto que el PBI ya esta por encima de sus niveles en la pandemia. 
# 
# Ahora bien, el autor mencionó tres formas de complementar la pregunta, estas son agregar la variable de agentes con expectativas racionales, proponer un modelo de estática comparativa más realista que considere también la respuesta pública en el ámbito económico. Finalmente, la creación de un modelo que explique los efectos de la duración de la pandemia en el producto potencial. 
# 
# Ahora bien, en mi opinión otras formas diferentes de ayudar a seguir con la pregunta de investigación sería ver qué  sucedió en otros países de la región en terminos macroeconomicos a raíz del COVID-19. Pues de esta forma se podría ver si la afectación fue similar o distinta y en base a ello evaluar posibles factores que logren explicarlo. En base a ello propongo el texto de Botero (2022), en donde se analiza a nivel macroeconómico los efectos del COVID-19 en la economía colombiana. 
# 
# Por otro lado, para entender mejor los efectos del COVID en el Perú, se podría analizar más el contexto económico peruano antes del inicio de la pandemia. En base a ello sería de mucha utilidad el texto de Barrutia, Silva y Sanchez (2021), en el cual en base a fuentes como el MEF y el MINSA se nos indica que la economía peruana ya se encontraba sufriendo fluctuaciones y retrocesos antes de la pandemia y que se terminó acelerando con la llegada de esta. Por otro parte, nos indican de que si bien el pbi está en recuperando, el trabajo formal sigue desplomado. 
# 
# Fuentes: 
# 
# Barrutia, I., Silva, H., Sánchez R. (2021). Consecuencias económicas y sociales de la inamovilidad humana bajo COVID-19: caso de estudio Perú 
# 
# http://www.scielo.org.co/scielo.php?script=sci_arttext&pid=S0120-25962021000100285
# 
# 
# Macroeconomía en los tiempos del Covid-19: un análisis de equilibrio dinámico estocástico para Colombia
# 
# https://repositorio.banrep.gov.co/bitstream/handle/20.500.12134/10311/2.Macroeconom%C3%ADa%20en%20los%20tiempos%20del%20Covid-19%20un%20an%C3%A1lisis%20de%20equilibrio%20din%C3%A1mico%20estoc%C3%A1stico%20para%20Colombia.pdf?sequence=1&isAllowed=y
# 
# 
# 
# Instituto Peruano de Economía (2021) Cómo impacta la minería en la productividad de Perú 
# 
# https://www.ipe.org.pe/portal/ipe-como-impacta-la-mineria-en-la-productividad-de-peru/
# 
# 
# 
# 
# 
# 
