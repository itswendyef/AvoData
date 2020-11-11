from joblib import dump, load

# Cargar el modelo predicho
model = load('sup_learn_avo.joblib')

# Valor a predecir

aguaEsperada = [
    25,   # arena
    40,      # limo
    35,      # arcilla
    3,      # materia_organica
    3,      # edad 
    2200,   # tamano
    2350,   # volumen_produccion
    0,      # tempertatura
    5,   # pendiente
    10,  # humedad 
]

# El metodo requiere de recibir un array

A_evaluar = [
    aguaEsperada
]

# Procesa el valor y devuelve un array con la predicción
procesado = model.predict(A_evaluar)

# Obtiene el valor predicho para hoy
agua_valorf = procesado[0]

print("Litros de agua para hoy: {:,.2f}".format(agua_valorf))

