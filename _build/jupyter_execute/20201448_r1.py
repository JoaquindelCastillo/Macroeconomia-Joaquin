#!/usr/bin/env python
# coding: utf-8

# # REPORTE Nº1
# 
# Nombre: Joaquin del Castillo Ugarte
# 
# Codigo: 20201448
# 
# 
# 

# Parte 1: Codigo

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from sympy import *
import pandas as pd
from causalgraphicalmodels import CausalGraphicalModel


# $$Pregunta: 1$$
# 
# Derivar y explicar paso a paso la función de demanda de consumo:
# 
# La formula de la demanda de consumo es la siguiente
# 
# 
# $$C= Co+bY^d$$
# 
# 
# Para sacar el ingreso disponible que es necesario para ver la formula de la demanda del consumo, primero se debe ver la tributación, la cual se obtiene de la siguiente fórmula: 
# 
# 
# $$T=tY$$ 
# 
# 
# Ahora de la resta del ingreso con la tributación se saca el ingreso disponible.
# 
# 
# $$Y^d=Y-T$$
# 
# 
# Ahora solo factorizamos: 
# 
# 
# $$Y^d=(1-t)Y$$
# 
# 
# Ya con el con el conocimiento del ingreso disponible se puede ver el consumo variable, el cual se halla de la multiplicación entre el ingreso disponible con la propensión marginal a consumir, la cual es el porcentaje del ingreso disponible que representa el consumo variable(bY^d). Con este dato finalmente se suma con el consumo fijo (Co), el cual no varía ni depende del ingreso variable. Ello nos da finalmente la función de la demanda del consumo: 
# 
# $$C=Co+bY^d$$
# 
# 

# In[2]:


# Parámetros

Y_size = 100 

Co = 10
b = 0.5 

Y = np.arange(Y_size)

# Ecuación de la curva del ingreso de equilibrio

def DA_A(Co, b):
    DA_A = (Co) + (b*Y)
    return DA_A

DA_A_A = DA_A(Co, b)


# In[3]:


# Recta de 45°

a = 2.5 

def L_45(a, Y):
    L_45 = a*Y
    return L_45

L_45 = L_45(a, Y)


# In[4]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(DA_A_A)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, DA_A_A, label = "C", color = "#e721c5")
# ax.plot(Y, L_45, color = "#404040") #Línea de 45º

# Título y leyenda
ax.set(title="Función de demanda de consumo", xlabel= r'Y', ylabel= r'C')
ax.legend() #mostrar leyenda

plt.show()


# $$Pregunta: 2$$
# 
# Ahora se derivara la función de Ingreso: I=Io-hr
# 
# En primer lugar veremos la inversión autonoma que viene a representar la varible autonoma, depende principalmente de las expectativas de los inversionistas: 
# 
# $$Io$$
# 
# Ahora bien, r viene a representar la tasa de interés, es decir cuanto es el interés que les cobran a los inversiones por los prestamos, mientras el r esté más cerca al 1, influenciará de manera más negativa a la inversión en el país, pues los inversionistas tendrán menos incentivos a invertir. Por ello se usa el signo (-) pues a mayor sea el r menor será la inversión. 
# 
# $$-r$$
# 
# FInalmente tenemos a h que es la constante de decisión a invertir, también varía entre 0 y 1, la cual interactua con la tasa de intereses, por lo que a medida que r sea menor, la inversión será mayor.
# 
# $$h$$
# 
# Finalmente, nos queda la siguiente ecuación, en donde la Inversión Autonoma es restada por la multiplicación de la tasa de interes con la constante de decisión a invertir, ello determinará finalmente, la demanda de inversión en el país: 
# 
# $$I=Io-hr$$    
# 

# In[5]:


# Parámetros

r_size = 100 

Io = 40
h =  0.7


r = np.arange(r_size)

# Ecuación de la curva del ingreso de equilibrio

def DA_B(Io, h):
    DA_B = (Io) - (h*r)#no se puede poner r porque sale error, asi que estoy asumiendo en este gráfico que Y=r
    return DA_B

DA_B_B = DA_B(Io, h)


# In[6]:


# Recta de 45°

a = 2.5 

def L_45(a, r):
    L_45 = a*r
    return L_45

L_45 = L_45(a, r)


# In[7]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(DA_B_B)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(r, DA_B_B, label = "I", color = "#59e31a") #Demanda agregada
# ax.plot(r, L_45, color = "#404040") #Línea de 45º

# Título y leyenda
ax.set(title="Función de demanda de inversión", xlabel= r'r', ylabel= r'I')
ax.legend() #mostrar leyenda

plt.show()


# $$Pregunta:3$$
# 
# Algunos supuestos importantes son:a)La tasa de interes se determina por fuera del modelo (por mercado monetario). b)El modelo es a corto 
# plazo. c)El nivel de producto cambia en función del cambio de la demanda agregada. d)El modelo presupone que los precios son fijos. 
# 
# Ahora, a continuación se explicarán algunos puntos importantes sobre los supuestos: 
# 
# - Las ecuaciones del modelo de Ingreso -Gasto y la función de Demanda Agregada:
# 
# -A continuación veremos como se construye la pendiente y el intercepto de la función DA, vital para entender los supuestos: 
# 
# Primero sabemos que la siguiente formula corresponde a la DA:
# 
# $$ DA = C + I + G + X - M $$
# 
# Ahora bien, la descomponemos obteniendo los datos necesarios para poder obtener la demanda de consumo, gasto publico, inversiones y 
# exportaciones menos importaciones:
# 
# $$DA=Co+bY^d+Io-hr+Go+Xo-mY^d$$
# 
# Ahora separamos las variables que presenten Y las factorizamos, mientras que a las demás las colocamos en otro lado:
# 
# $$ DA = C_0 + I_0 + G_0 + X_0 - hr + Y(b - m)(1 - t) $$
# 
# Listo, ahora ya hemos separado al intercepto de la pendiente: el intercepto será α_0, mientras que la pendiente será α_1Y
# 
# $$ DA = α_0 + α_1Y $$

# a)La determinación del Ingreso de equilibrio: 
# 
# En equilibrio la Demanda Agregada (DA) debe ser igual al ingreso agregado Y donde se obtiene la siguiente formula de equilibrio de producción a corto plazo. 
# 
# Debido a que DA=Y, es igual el ingreso con la formula de la DA
# 
# $$ Y = C_0 + bY^d + I_0 -hr + G_0 + X_0 - mY^d $$ 
# 
# Ahora bien, de esta formula, finalmente simplificamos la ecuación a la siguiente: 
# 
# $$ Y = \frac{1}{1 - (b - m)(1 - t)} (C_0 + I_0 + G_0 + X_0 - hr) $$
# 
# A corto plazo se supone que los precios no son elasticos, sino fijos. MIentras que la oferta agregada es muy elastica en función a cada precio. En este sentido la oferta está en función de la demanda agregada. 

# b) La estabilidad de equilibrio: La producción o Y se mueve en función de la demandada agregada. Es decir que si la demanda crece, las empresas al ver ello produciran más bienes y de esa forma se ve como la producción se nivelara. Por otro lado, si la demanda cae, las empresas tenderán a producir menos. Entonces, se ve que el equilibrio se produce en función de como se mueva la demanda agregada. 

# c)El multiplicador keynesiano: El multiplicador keynesiano depende de las variables t, b y m, pues viene a representar su denominador. Además, un cambio de las variables autonomas de demanda agregada generan un cambio multiplicado en Y.
# 

# $$Pregunta: 4$$
# A continuación se muestra la ecuación de la demanda agregada: 
# 
# $$ DA = C + I + G + X - M $$
# 
# Esta formula a su vez puede descomponerse en las siguientes variables: 
# 
# $$DA=Co+bY^d+Io-hr+Go+Xo-mY^d$$
# 
# Ahora bien, gracias a la teoría sabemos que el ingreso de equilibrio es cuando el ingreso y la demanda agregada son iguales. Por ende, en este caso igualaremos el ingreso con la demanda, teniendo como variable dependiente al ingreso. 
# 
# 
# $$ Y = C_0 + bY^d + I_0 -hr + G_0 + X_0 - mY^d $$ 
# 
# Ahora bien, ya con la siguiente formula solo queda agrupar a nuestro intercepto por un lado, mientras que en el otro las variables que cuentan con la Y, la cual finalmente reemplazaremos y nos quedaría al final la siguiente fórmula de equilibrio: 
# 
# $$ Y = \frac{1}{1 - (b - m)(1 - t)} (C_0 + I_0 + G_0 + X_0 - hr) $$

# In[8]:


# - Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8 # b > m
m = 0.2
t = 0.3
r = 0.9

Y = np.arange(Y_size)

# Ecuación de la curva del ingreso de equilibrio

def DA_C(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_C = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_C

DA_C_C = DA_C(Co, Io, Go, Xo, h, r, b, m, t, Y)


# In[9]:


# Recta de 45°

a = 2.5 

def L_45(a, Y):
    L_45 = a*Y
    return L_45

L_45 = L_45(a, Y)


# In[10]:


#Gráfico

# Dimensiones del gráfico
y_max = np.max(DA_C_C)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, DA_C_C, label = "DA", color = "#02ffea") #Demanda agregada
ax.plot(Y, L_45, color = "#ff00a7") #Línea de 45º

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Líneas punteadas punto de equilibrio
plt.axvline(x=70.5,  ymin= 0, ymax= 0.69, linestyle = ":", color = "grey")
plt.axhline(y=176, xmin= 0, xmax= 0.7, linestyle = ":", color = "grey")

# Texto agregado
   # punto de equilibrio
plt.text(0, 220, '$DA=α_o+α_1$', fontsize = 11.5, color = '#2238d3')
plt.text(72, 0, '$Y^e$', fontsize = 12, color = '#2238d3')
plt.text(0, 180, '$α_o=Co+Io+Go+Xo-hr$', fontsize = 15, color = '#2238d3')
plt.text(72, 130, '$α_1=(b-m)(1-t)$', fontsize = 15, color = '#2238d3')
   # línea 45º
plt.text(6, 4, '$45°$', fontsize = 11.5, color = '#2238d3')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#2238d3')

# Título y leyenda
ax.set(title="El ingreso de Equilibrio a Corto Plazo", xlabel= r'Y', ylabel= r'DA')
ax.legend() #mostrar leyenda

plt.show()


# $$Pregunta: 5$$
# 
# A) 
# 
# - Intuición: 
# 
# $$ Go↑ → G↑ → DA↑ → DA > Y → Y↑ $$
# 
# 
# 
# 

# - Matemáticamente: $∆Go > 0  →  ¿∆Y?$

# $$ Y = \frac{1}{1 - (b - m)(1 - t)} (C_0 + I_0 + G_0 + X_0 - hr) $$
# 
# o, considerando el multiplicador keynesiano, $ k > 0 $:
# 
# $$ Y = k (C_0 + I_0 + G_0 + X_0 - hr) $$
# 
# 
# $$ ∆Y = k (∆C_0 + ∆I_0 + ∆G_0 + ∆X_0 - ∆hr) $$

# Pero, si no ha habido cambios en $C_0$, $I_0$, $X_0$, $h$ ni $r$, entonces: 
# 
# $$∆C_0 = ∆I_0 = ∆X_0 = ∆h = ∆r = 0$$
# 
# $$ ∆Y = k (∆G_0) $$

# $$ ∆Y = (+)(+) $$
# 
# $$ ∆Y > 0 $$

# In[11]:


#--------------------------------------------------
# Curva de ingreso de equilibrio ORIGINAL

    # Parámetros
Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3
r = 0.9

Y = np.arange(Y_size)

    # Ecuación 
def DA_C(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_C = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_C

DA_C_C = DA_C(Co, Io, Go, Xo, h, r, b, m, t, Y)


#--------------------------------------------------
# NUEVA curva de ingreso de equilibrio

    # Definir SOLO el parámetro cambiado
Go = 100

# Generar la ecuación con el nuevo parámetro
def DA_C(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_C = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_C

DA_G = DA_C(Co, Io, Go, Xo, h, r, b, m, t, Y)


# In[12]:


# Gráfico
y_max = np.max(DA_C_C)
fig, ax = plt.subplots(figsize=(10, 8))


# Curvas a graficar
ax.plot(Y, DA_C_C, label = "DA", color = "#19e1a8") #curva ORIGINAL
ax.plot(Y, DA_G, label = "DA_G", color = "#EE7600", linestyle = 'dashed') #NUEVA curva
ax.plot(Y, L_45, color = "#c02bca") #línea de 45º

# Lineas punteadas
plt.axvline(x = 70.5, ymin= 0, ymax = 0.69, linestyle = ":", color = "grey")
plt.axhline(y = 176, xmin= 0, xmax = 0.7, linestyle = ":", color = "grey")
plt.axvline(x = 84,  ymin= 0, ymax = 0.82, linestyle = ":", color = "grey")
plt.axhline(y = 210.5, xmin= 0, xmax = 0.82, linestyle = ":", color = "grey")

# Texto agregado
plt.text(0, 180, '$DA^e$', fontsize = 11.5, color = '#870ff3')
plt.text(0, 125, '$DA_G$', fontsize = 11.5, color = '#870ff3')
plt.text(6, 4, '$45°$', fontsize = 11.5, color = '#870ff3')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')
plt.text(72, 0, '$Y^e$', fontsize = 12, color = 'black')
plt.text(56, 0, '$Y_G$', fontsize = 12, color = '#EE7600')
plt.text(78, 45, '$→$', fontsize = 18, color = '#ee1abb')
plt.text(20, 165, '$↑$', fontsize = 18, color = '#ee1abb')

# Título y leyenda
ax.set(title = "Aumento del Gasto del Gobierno $(G_0)$", xlabel = r'Y', ylabel = r'DA')
ax.legend()

plt.show()


# B)
# 
# - Intuición: ¿contradicción?
# 
# $$ t↓ → Co↑ → DA↑ → DA > Y → Y↑ $$
# $$ t↓ → M↑ → DA↓ → DA < Y → Y↓ $$

# - Matemáticamente:
#  
#  $$∆t < 0  →  ¿∆Y?$$

# In[13]:



Co, Io, Go, Xo, h, r, b, m, t = symbols('Co Io Go Xo h r b m t')

f = (Co + Io + Go + Xo - h*r)/(1-(b-m)*(1-t))


df_t = diff(f, t)
df_t #∆Y/∆t


# Considernado el diferencial de $∆t$:
# 
# $$ \frac{∆Y}{∆t} = \frac{(m-b)(Co + Go + Io + Xo - hr)}{(1-(1-t)(b-m)+1)^2} $$
# 
# - Sabiendo que b > m, entonces $(m-b) < 0$
# 
# - Los componentes autónomos no cambian: $∆C_0 = ∆I_0 = ∆X_0 = ∆h = ∆r = 0$
# 
# - Cualquier número elevado al cuadrado será positivo: $ (1-(1-t)(b-m)+1)^2 > 0 $
# 
# Entonces:
# 
# $$ \frac{∆Y}{∆t} = \frac{(-)}{(+)} $$
# 
# Dado que $∆t < 0$, entonces se asume que toma un valor -, el cual pasa a multiplicar al otro - entonces (-) (-) da finalmente + y ya que es divido por un valor positivo da finalmente un número + :
# 
# $$ \frac{∆Y}{(-)} = \frac{(-)}{(+)} $$
# 
# $$ ∆Y = \frac{(-)(-)}{(+)} $$
# 
# $$ ∆Y > 0 $$
# 
#  
# 

# - Grafíco: 

# In[14]:


#--------------------------------------------------
# Curva de ingreso de equilibrio ORIGINAL

    # Parámetros
Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3 #tasa de tributación
r = 0.9

Y = np.arange(Y_size)

    # Ecuación 
def DA_D(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_D = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_D

DA_D_D = DA_D(Co, Io, Go, Xo, h, r, b, m, t, Y)


#--------------------------------------------------
# NUEVA curva de ingreso de equilibrio

    # Definir SOLO el parámetro cambiado
t = 0.01

# Generar la ecuación con el nuevo parámetros
def DA_D(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_D = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_D

DA_t = DA_D(Co, Io, Go, Xo, h, r, b, m, t, Y)


# In[15]:


# Gráfico
y_max = np.max(DA_D_D)
fig, ax = plt.subplots(figsize=(10, 8))


# Curvas a graficar
ax.plot(Y, DA_D_D, label = "DA", color = "#3D59AB") #curva ORIGINAL
ax.plot(Y, DA_t, label = "DA_t", color = "#EE7600", linestyle = 'dashed') #NUEVA curva
ax.plot(Y, L_45, color = "#c02bca") #línea de 45º

# Lineas punteadas
plt.axvline(x = 70.5, ymin= 0, ymax = 0.69, linestyle = ":", color = "grey")
plt.axhline(y = 176, xmin= 0, xmax = 0.7, linestyle = ":", color = "grey")
plt.axvline(x = 77,  ymin= 0, ymax = 0.75, linestyle = ":", color = "grey")
plt.axhline(y = 192, xmin= 0, xmax = 0.75, linestyle = ":", color = "grey")

# Texto agregado
plt.text(0, 180, '$DA^e$', fontsize = 11.5, color = 'black')
plt.text(0, 200, '$DA_t$', fontsize = 11.5, color = '#EE7600')
plt.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')
plt.text(72, 0, '$Y^e$', fontsize = 12, color = 'black')
plt.text(80, 0, '$Y_t$', fontsize = 12, color = '#EE7600')
plt.text(72, 45, '$→$', fontsize = 18, color = 'grey')
plt.text(20, 180, '$↑$', fontsize = 18, color = 'grey')

# Título y leyenda
ax.set(title = "Reducción de la Tasa de Tributación", xlabel = r'Y', ylabel = r'DA')
ax.legend()

plt.show()


# En la gráfica también se cumple que ∆Y > 0. Por tanto en las tres formas se cumple que ∆Y > 0 cuando la reducción de la tasa de tributación baja. 

# $$Pregunta: 6$$

# A)

# In[16]:


# - Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8 # b > m
m = 0.2
t = 0.3
r = 0.9
g = 0.4

Y = np.arange(Y_size)

# Ecuación de la curva del ingreso de equilibrio

def DA_E(Co, Io, Go, Xo, h, r, b, m, t, g,Y):
    DA_E = (Co + Io + Go + Xo - h*r) + (((b - m)*(1 - t)- g)*Y)
    return DA_E

DA_E_E = DA_E(Co, Io, Go, Xo, h, r, b, m, t, g, Y)


# In[17]:


#Gráfico

# Dimensiones del gráfico
y_max = np.max(DA_E_E)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, DA_E_E, label = "DA", color = "#c94cb7") #Demanda agregada
ax.plot(Y, L_45, color = "#22ec0a") #Línea de 45º

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Líneas punteadas punto de equilibrio
plt.axvline(x=59,  ymin= 0, ymax= 0.58, linestyle = ":", color = "grey")
plt.axhline(y=147, xmin= 0, xmax= 0.7, linestyle = ":", color = "grey")

# Texto agregado
    # punto de equilibrio
plt.text(0, 220, '$DA=α_o+α_1$', fontsize = 11.5, color = '#e33710')
plt.text(72, 0, '$Y^e$', fontsize = 12, color = '#e33710')
plt.text(0, 180, '$α_o=Co+Io+Go+Xo-hr$', fontsize = 15, color = '#e33710')
plt.text(68, 130, '$α_1=(b-m)(1-t)-g$', fontsize = 15, color = '#e33710')
    # línea 45º
plt.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')

# Título y leyenda
ax.set(title="El ingreso de Equilibrio a Corto Plazo con política contraciclica", xlabel= r'Y', ylabel= r'DA')
ax.legend() #mostrar leyenda

plt.show()


# B) 
# 
# Primero vemos la formula de la demanda agregada:
# 
# $$ DA = C + I + G + X - M $$
# 
# -Luego de ello desglosamos, vemos que en este caso con una política fiscal con regla contraciclica, por lo que agregamos un -g en Gobierno. Por tanto nos sale lo siguiente. 
# 
# $$𝐷𝐴 = (𝐶o + 𝐼o + 𝐺o + 𝑋o − ℎ𝑟) + ((𝑏 − 𝑚)(1 − 𝑡) − 𝑔)*Y$$
# 
# Ahora tras efectuar el proceso aritmetico, nos sale lo siguiente: 
# 
# $$ Y = \frac{1}{1 - (b - m)(1 - t)+g} (C_0 + I_0 + G_0 + X_0 - hr) $$

# C)
# 
# - Inferencial: 
# 
# $$ G↑ → Go↑ → (Go-gY)↑ → DA↑ → DA > Y → Y ↑$$
# 

# - Matemáticamente: 
#     
#  $∆Go > 0  →  ¿∆Y?$
# 
# $$ Y = \frac{1}{1 - (b - m)(1 - t)+g} (C_0 + I_0 + G_0 + X_0 - hr) $$
# 
# o, considerando el multiplicador keynesiano, $ k > 0 $:
# 
# $$ Y = k (C_0 + I_0 + G_0 + X_0 - hr) $$
# 
# 
# $$ ∆Y = k (∆C_0 + ∆I_0 + ∆G_0 + ∆X_0 - ∆hr) $$
# 
# 

# Pero, si no ha habido cambios en $C_0$, $I_0$, $X_0$, $h$ ni $r$, entonces: 
# 
# $$∆C_0 = ∆I_0 = ∆X_0 = ∆h = ∆r = 0$$
# 
# $$ ∆Y = k (∆G_0) $$

# Sabiendo que $∆G_0 > 0 $ y que $k > 0$, la multiplicación de un número positivo con un positivo dará otro positivo:
# 
# $$ ∆Y = (+)(+) $$
# 
# $$ ∆Y > 0 $$

# - Gráfico: 

# In[18]:


# - Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8 # b > m
m = 0.2
t = 0.3
r = 0.9
g = 0.4

Y = np.arange(Y_size)

# Ecuación de la curva del ingreso de equilibrio

def DA_F(Co, Io, Go, Xo, h, r, b, m, t, g,Y):
    DA_F = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)- g)*Y
    return DA_F

DA_F_F = DA_F(Co, Io, Go, Xo, h, r, b, m, t, g, Y)

#--------------------------------------------------
# NUEVA curva de ingreso de equilibrio

    # Definir SOLO el parámetro cambiado
Go = 100

# Generar la ecuación con el nuevo parámetro
def DA_F(Co, Io, Go, Xo, h, r, b, m, t, g, Y):
    DA_F = (Co + Io + Go + Xo - h*r) + (((b - m)*(1 - t)-g)*Y)
    return DA_F

DA_G = DA_F(Co, Io, Go, Xo, h, r, b, m, t, g, Y)


# In[19]:


# Gráfico
y_max = np.max(DA_F_F)
fig, ax = plt.subplots(figsize=(10, 8))


# Curvas a graficar
ax.plot(Y, DA_F_F, label = "DA", color = "#3D59AB") #curva ORIGINAL
ax.plot(Y, DA_G, label = "DA_G", color = "#6ee310", linestyle = 'dashed') #NUEVA curva
ax.plot(Y, L_45, color = "#e33710") #línea de 45º

# Lineas punteadas
plt.axvline(x = 70.5, ymin= 0, ymax = 0.69, linestyle = ":", color = "grey")
plt.axhline(y = 176, xmin= 0, xmax = 0.7, linestyle = ":", color = "grey")
plt.axvline(x = 59,  ymin= 0, ymax = 0.60, linestyle = ":", color = "grey")
plt.axhline(y = 148, xmin= 0, xmax = 0.72, linestyle = ":", color = "grey")

# Texto agregado
plt.text(0, 180, '$DA^e$', fontsize = 11.5, color = 'black')
plt.text(0, 125, '$DA_G$', fontsize = 11.5, color = '#EE7600')
plt.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')
plt.text(72, 0, '$Y^e$', fontsize = 12, color = 'black')
plt.text(56, 0, '$Y_G$', fontsize = 12, color = '#EE7600')
plt.text(60, 45, '$→$', fontsize = 18, color = 'grey')
plt.text(20, 160, '$↑$', fontsize = 18, color = 'grey')

# Título y leyenda
ax.set(title = "Aumento del Gasto del Gobierno con política contraciclica $(G_0)$", xlabel = r'Y', ylabel = r'DA')
ax.legend()

plt.show()


# Lo que se ve es que el ingreso aumenta, en otras palabras, ∆Y > 0. Por tanto se cumple que ∆Y > 0 en las tras formas. 

# D) El g lo que esta haciendo es sumar el denominador del multiplicador keynesiano, por lo tanto esta aumentando el denominador y con ello reduciendo el valor del multiplicador keynesiano. 

# E) No el efecto no es el mismo. Pues si bien en ambos subió el Go en la misma magnitud. En el segundo caso al tratarse de una política contraciclica se resta el Go-gY, es decir que si bien el Go crece igual, en la segunda política el crecimiento es menor, pues el g disminuye el gasto. Entonces, si bien ambos casos experimentan una subida del gasto en general, el segundo caso tiene un crecimiento menor, pues tiene a la variable g que resta. Por ende en el primer caso la Y crece más que en el segundo, pues se aplica una política contraciclica. 

# F) 
# - Inferencial: 
# 
# $$ Xo↓ → X↓ → DA↓ → DA < Y → Y↓ $$
# 
# 
# 

# - Matemático: 
# 
#  Matemáticamente: $∆Xo < 0  →  ¿∆Y?$
# 
# $$ Y = \frac{1}{1 - (b - m)(1 - t)+g} (C_0 + I_0 + G_0 + X_0 - hr) $$
# 
# o, considerando el multiplicador keynesiano, $ k > 0 $:
# 
# $$ Y = k (C_0 + I_0 + G_0 + X_0 - hr) $$
# 
# 
# $$ ∆Y = k (∆C_0 + ∆I_0 + ∆G_0 + ∆X_0 - ∆hr) $$
# 
# 
# 

# Pero, si no ha habido cambios en $C_0$, $I_0$, $G_0$, $h$ ni $r$, entonces: 
# 
# $$∆C_0 = ∆I_0 = ∆G_0 = ∆h = ∆r = 0$$
# 
# $$ ∆Y = k (∆X_0) $$

# Sabiendo que $∆X_0 < 0 $ y que $k > 0$, la multiplicación de un número negativo con un positivo dará otro negativo:
# 
# $$ ∆Y = (-)(+) $$
# 
# $$ ∆Y < 0 $$

# - Gráfico: 

# In[20]:


#Gráfico: 

# - Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 45
h = 0.7
b = 0.8 # b > m
m = 0.2
t = 0.3
r = 0.9
g = 0.4

Y = np.arange(Y_size)

# Ecuación de la curva del ingreso de equilibrio

def DA_G(Co, Io, Go, Xo, h, r, b, m, t, g,Y):
    DA_G = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)- g)*Y
    return DA_G

DA_G_G = DA_G(Co, Io, Go, Xo, h, r, b, m, t, g, Y)

#--------------------------------------------------
# NUEVA curva de ingreso de equilibrio

    # Definir SOLO el parámetro cambiado
Go = 10

# Generar la ecuación con el nuevo parámetro
def DA_G(Co, Io, Go, Xo, h, r, b, m, t, g, Y):
    DA_G = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)-g)*Y
    return DA_G

DA_X = DA_G(Co, Io, Go, Xo, h, r, b, m, t, g, Y)


# In[21]:


#Gráfico
y_max = np.max(DA_F_F)
fig, ax = plt.subplots(figsize=(10, 8))


# Curvas a graficar
ax.plot(Y, DA_G_G, label = "DA", color = "#3D59AB") #curva ORIGINAL
ax.plot(Y, DA_X, label = "DA_X", color = "#EE7600", linestyle = 'dashed') #NUEVA curva
ax.plot(Y, L_45, color = "#6ee310") #línea de 45º

# Lineas punteadas
plt.axvline(x = 76.5, ymin= 0, ymax = 0.75, linestyle = ":", color = "grey")
plt.axhline(y = 189.5, xmin= 0, xmax = 0.75, linestyle = ":", color = "grey")
plt.axvline(x = 51.5,  ymin= 0, ymax = 0.54, linestyle = ":", color = "grey")
plt.axhline(y = 130, xmin= 0, xmax = 0.54, linestyle = ":", color = "grey")

# Texto agregado
plt.text(0, 180, '$DA^e$', fontsize = 11.5, color = 'black')
plt.text(0, 121, '$DA_G$', fontsize = 11.5, color = '#EE7600')
plt.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
plt.text(2.5, -3, '$◝$', fontsize = 30, color = '#404040')
plt.text(72, 0, '$Y^e$', fontsize = 12, color = 'black')
plt.text(56, 0, '$Y_G$', fontsize = 12, color = '#EE7600')
plt.text(60, 45, '$←$', fontsize = 18, color = 'grey')
plt.text(20, 160, '$↓$', fontsize = 18, color = 'grey')

# Título y leyenda
ax.set(title = "Reducción de las exportaciones del Gobierno con política contraciclica $(G_0)$", xlabel = r'Y', ylabel = r'DA')
ax.legend()

plt.show()


# Lo que se ve entonces en la gráfica es que ∆Y < 0. Por tanto se cumple en las tres formas que ∆Y < 0. 

# Parte 2: Reporte
# 
# La pregunta de investigación en mi opinión debería ser la siguiente: ¿A qué se debe la desaceleración económica sufrida en el Perú en el periodo de 2013-2015 y cómo el Estado y el BCR respondieron frente a esta crisis? Ahora bien, con respecto a las fortalezas se puede identificar la organización con la cual presenta los hechos. Pues primero, nos narra los factores externos que pueden explicar la crisis, por ejemplo, que al haber una caída internacional de los precios de los metales, las inversiones en la minería iban a ser menores, lo cual iba a ser muy dañino para el Perú. Mostrando, en segundo lugar, como el mal manejo interno derivó en, según el autor, la no recuperación de la economía. Llegando finalmente, a las conclusiones en donde el autor no solo se limita a mencionar los errores cometidos en este periodo de tiempo, sino que también brinda recomendaciones que podría seguir, el para entonces futuro presidente del Perú, Pedro Pablo Kuczynski. 
# 
# Por otro lado, considero que el paper si bien es muy interesante y bueno para entender mejor los problemas de la economía peruana, en ocasiones no da mucha explicación sobre algunos términos técnicos de economía, que si bien no son un problema para entender el documento en general, ocasionan que en partes de la explicación sea complicado seguir la ilación de argumento, por ejemplo ello se ve en la introducción del concepto de la tasa de morosidad, la cual no fue explicada y que su comprensión era vital si se quería entender qué era el efecto hoja de balance. Por otro lado, considero que al abordar tantos temas (impacto externo en la economía, inflación y tipo de cambio, y como afecta en el Perú políticas monetarias y fiscales en 2013-2015, recomendaciones de cara al futuro) se requería un mayor número de páginas si se quería profundizar en los conceptos.
# 
# Ahora bien, sobre la contribución que realiza a la academia, me parece que es principalmente su enfoque keynesiano sobre cómo se podría haber llevado mejor el periodo de vacas flacas entre 2013-2015. El autor identifica algunos puntos muy importantes, por ejemplo, que en este periodo lo idóneo hubiera sido que el Estado invierta mucho más en obras públicas para reducir el desempleo y con ello la crisis. Además de mencionar que es necesario que se bajen las tasas de interés, como también es importante tener una buena reserva en dólares para vender en estas épocas de vacas flacas y no como como hizo el BCR que gastó gran parte de los dólares del Estado, lo cual lo dejaba en una situación muy compleja de cara a enfrentar futuras crisis y más teniendo en cuenta que la economía del Perú está muy dolarizada, es decir que gran parte de nuestros productos son importados o parte de sus insumos provienen de afuera, lo cual significa que ante un aumento del dólar estos también lo harán. En mi opinión al identificar este tipo de problemas permite al lector entender, según la visión keynesiana, cual fue el principal error que hubo en el Perú para enfrentar dicha crisis. 
# 
# Por otro lado, también considero que explican bien cómo funciona la inflación en el Perú, mencionando por ejemplo que la inflación siempre viene acompañada de una depreciación del sol frente al dólar, en consecuencia de ser la economía peruana una economía dolarizada. También, por ejemplo, cuestiona la política seguida por el Perú de mantener las metas de inflación en cifras tan pequeñas (una tasa del 2%), pues es muy difícil permanecer en ese margen, lo cual fue comprobado por el autor al demostrar el porcentaje del tiempo durante distintos periodos en el que la economía del país se mantuvo en esa cifra, siendo este muy pequeño, lo cual fue cuestionado por el autor mencionando que ello genera que las empresas no planifiquen sus economías en función de la meta inflacionaria del país, además, que podría ser otro factor que agrave más una crisis inflacionaria. 
# 
# Finalmente, considero que, en primer lugar, si se desea probar una cuestión tan grande como que las crisis inflacionarias o de “vacas flacas” son más evitables mediante una mayor inversión estatal, reducción de tasas de interés como también de una reserva en dólares para utilizar en momentos de crisis; debería mostrar casos externos, con la intención de fortalecer su argumento. Pues, durante todo el artículo, en la mayoría del tiempo justifica sus enunciados citando el ejemplo peruano en la crisis del 2008-2009. Si bien lo considero un dato importante, en mi opinión, ganaría más peso su argumento mostrando también ejemplos externos.En este sentido, en el caso de la autora Cabezas (2011), estudia en Ecuador las crisis vividas bajo justamente la óptica keynesiana. Teniendo en cuenta que este país es similar al Perú, en cuestiones como pertenencia a la misma región y un desarrollo medianamente similar, una comparación de ambos casos con la intención de fortalecer el argumento keynesiano sobre cómo salir de las crisis económicas sería de mucha utilidad. 
# 
# Por otro lado, considero que también sería muy interesante y enriquecedor comparar esta propuesta con otras distintas que expliquen también la razón de la época de vacas flacas del Perú de 2013 a 2015 y como lograr salir de ella. Por ejemplo, la visión de la escuela de Chicago, la cual cuestiona el papel del Estado en la economía, al igual que en muchas otras áreas. En ese sentido un texto que ayuda en esta dirección es el de Zamarriego (2014), pues este mediante la aplicación de las ideas de Friedman, analiza tendencias vividas en países como España o Francia en épocas de crisis, llegando a la conclusión que se suele seguir modelos de aumentar del gasto, lo cual, según el autor es nocivo para la economía pues incrementa en corto plazo el déficit fiscal y a largo plazo genera una vertiginosa subida de la deuda pública, lo cual tiene severos impactos en el PBI. 
#  
# Fuentes: 
#  
# Cabezas, M. (2011). El gasto público y el crecimiento económico de Ecuador desde una perspectiva kaynesiana para el periodo del 2000-2008. [Tesis de bachillerato]. Universidad Nacional Politécnica. Repositorio:
#  
# https://bibdigital.epn.edu.ec/bitstream/15000/4361/1/CD-3957.pdf
# 
# 
#  
# Zamarriego, F. (2014). La matriz del gasto de Milton Friedman. El papel que el Estado juega en la distribución de la riqueza a través del gasto público. [Tesis de Bachillerato]. Universidad Pontificia Comillas. Repositorio Comillas: 
#  
# https://repositorio.comillas.edu/xmlui/handle/11531/95 
