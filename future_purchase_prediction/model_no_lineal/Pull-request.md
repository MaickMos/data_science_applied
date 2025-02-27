# Push Notificactions  module #3
En esta PR se hara entrega de un modelo predictivo para posibles personas interesadas en un producto. Dado un usuario y producto, se predice si compraria ese producto si estuvieran comprando en ese momento. Condiciones: los usuarios deben ser aquellos que realizan una compra de minimo 5 productos. Usar modelo lineal.  

## Preparacion del dataset
Apartir del dataset feature_frame se cargaron los datos y se realizaron las modificaciones:

### Filtro para compradores de mas de 5 productos
Se realizo un filtrado para aquellos clientes que en una misma orden_id hallan comprado igual o mas de 5 productos, permitiendo trabajar el modelo solo con clientes que hallan comprado la cantidad de productos indicadas.

* Apartir de esta division se trabaja con un total de 2.163.953 filas donde el 1.4% son la clase objetivo
### Division del dataset
Se intentaron dos formas de division:
1. De manera aleatoria, 10% datos para el testeo, aproximadamente 20% para la validacion y un aprox. 80% para entrenamiento.
    Esta forma de de division de los datos presenta el problema de que el modelo estara aprendiendo con datos o tendencias del futuro. Factores que directamente afectan al rendimiento del modelo cuando este en produccion. Por esta razon se decidio no implementarla.  

2. De manera temporal, donde esta seccionado de la siguente manera:
 * Entrenamiento 70% -   2020-10-05 hasta 2021-02-04
 * Testeo 20%: 2021-02-04 hasta 2021-02-22
 * Validacion 10%: 2021-02-22 hasta 2021-03-03
    Como se observa en la grafica de comportamiento de numero de ventas, el comportamiento del mercado presenta un cambio en la forma de comprar de los usuario, afectando directamente en las posibles predicciones del modelo.
    ![graficas](https://github.com/user-attachments/)

### Eliminacion de varaibles
Se eliminaron varaibles que no aportaban al modelo o que de ser implementadas, aumentaban innecesariamente la complejidad sin tener un rendimiento proporcional.
* Variant_Id: ID del producto
* user_id: ID del producto
* created_at: Fecha de creacion de la orden de compra
* order_id: Numero de la orden hecha
* product_type: categoria del producto
* order_date: Fecha y hora de la compra
* vendor: Nombre del proovedor

### Normalizacion
Para realizar la estandarizacion de los datos se implemento el StandarScaler(), usando la media y la desviacion estandar.

## Metricas de evaluacion
### Recall
Dado que el dataset es desbalanceado, siempre que se evalue el modelo tendra un porcentaje alto, por esta razon se usara principalmente la metrica de recall, permitiendo medir la proporcion de verdaderos positivos.
### precision-recall curve
Por otro lado para una mejor comprension y comparacion visual se implementa la curva de precision-recall, que compara la precision con el recall.
### ROC Curve
Es otra curva la cual permite medir el recall o True positive Rate (TPR) y el False positive Rate. Es curva mostrara un mejor rendimiento cuando la grafica se acerque a la esquina superior izquierda (1.0,0.0), permitiendo comparar facilmente entre modelos.

## Modelos
### Baseline
Primero se implemento un modelo basico sin usar algoritmos de machine learning. A paritr de metricas ya existentes en el dataset se realizaron las predicciones de compra de cada producto apartir de la variable goblal_popularity. Se implemento una predicion binaria con un umbral de 0.5.
Realizando una preddicion basica con estos parametro obtenemos un recall de 0.0%, sin embargo la grafica de precision-recall muestra un comportamiento aceptable en las prediciones. Como se puede ver en la siguente grafica. En la grafica ROC curve obtiene un valor del 0.79, siendo aceptable. Estableciendose como un buen baseline para comparar el rendimiento de otros modelos.
### Logistic Regression
Se implemento este modelo con los parametros por defecto para evidenciar su rendimiento. Sin ningun tipo de regulacion, sin algoritmo de optimizacion, numero de iteraciones de 500 y ajuste para clases desbalanceadas.
#### Train
Los resultados para las prediciones realizadas con el dataset train. Se obtuvo una metrica de recall del 63%. En la grafica precision recall-curve esta grafica se muestra por encima del baseline, evidenciando que tiene mejor comportamiento. Por otro en la ROC curve se tiene un valor de 0.83. Confirmando la superioridad en a el baseline. Mostrando que ha aprendido de los datos.
#### Test
Sin embargo al evaluar el modelo en con dataset test se evidencia que no ha sido capaz de generalizar las relaciones de los datos, mostrando un recall de 6%, en la grafica precision-recall esta por debajo de baseline muy cercano a cero y por ultimo en ROC tenemos un valor de 0.6, inferior a baseline. Por lo tanto este modelo no es eficiente.


### Logistic Regression Ridge
Se entreno nuevamente el modelo aplicandole una regulacion Ridge de 0.1. Se evidenciaron los mismos resultados que el modelo original tanto en test como en train. se grafico el comportamiento del train y el test comparado con el baseline, como se puede ver en la siguente grafica:

![graficas](https://github.com/user-attachments/)


Seguidamente se grafico el comportamiento frente a otro valores de regulacion, usando la funcion plot_metrics() desde 1000 hasta 1e-8 en potencias de 100.

En el entrenamiento:
Se puede ver un comportamiento similar a el modelo base, teniendo un valor que cambia la dinamica c=0.000001, mostrando un mejor desempeño en ciertas areas.
![graficas](https://github.com/user-attachments/)

En el test:
El modelo no demostro un rendimiento bueno, estando aun debajo de Baseline. De nuevo el modelo de C=0.0000001 tiene un mejor rendimiento.

![graficas](https://github.com/user-attachments/)

### Logistic Regression Lasso
Se realizaron las mismas pruebas con la regulacion de Lasso. Donde para de manera general se obtienen mejores resultados superando al baseline. En especifico el modelo de mayor con rendimiento es regulacion lasso C = 0.0001 con un AUC en la grafica ROC de 0.83. 

![graficas](https://github.com/user-attachments/)

## Pesos de las variables
Usando el modelo de mejor rendimeinto y dado que la regulacion Lasso envia mas coeficientes a ceros y da importancias a otras varaibles, se miraron las principales variables que aportan al modelo. teniendo que las mas importantes son:
* ordered_before
* global_popularity
* abandoned_before
Las demas varaibles tienen valor de cero, por lo tanto no aportan al entrenamiento del modelo. Y es posible presendir de estas.

## Product type
Por ultimo se realizo un entramiento con las tres varaibles que mas aportan al modelo y se implmento la varaible de product type con categorical encoding. Esto para comprobar si se obtiene un mejor rendimiento usando esta variable que presenta una alta cardinalidad teniendo 62 valores unicos.
Se realiza la preparacion del dataset, entrenamiento del modelo regresion logistica con regulacion Lasso y ridge de C=0.0001 y se comparo con este mismo modelo sin implemtar la variable de producty type.
![graficas](https://github.com/user-attachments/)

En la grafica Ridge muestra un comportamiento mejor que Baseline, sin embargo los que presentan un mayor desempeño son Lasso, mostrando el mismo comportamiento aun que se haya entrenado con categorical type. Se observa el peso de las categorias y dado que Lasso baja los coeficientes bajos a cero, nuevamente solo estan presentes las tres mismas variables.

Por ultimo se guarda el modelo Lasso con C=0.0001 con las tres variables mencionadas anteriormente.

# Archivo para cargar el modelo
Se creo el script load_model.py, al permite realizar la carga del modelo y del normalizador.
Presetan las funciones