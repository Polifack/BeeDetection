using Statistics
using Flux: onehotbatch, onecold, crossentropy, binarycrossentropy
using Flux
using Random
using DelimitedFiles

include("funciones_utiles.jl");

ArquitecturaRNA = [5, 4];
PorcentajeValidacion = 0.2;
PorcentajeTest = 0.2;
NumMaxCiclosEntrenamiento = 1000;
NumMaxCiclosSinMejorarValidacion = 100;

numEjecuciones = 50;

# Dejar esta variable como:
#  "setosa", "virginica" o "versicolor" para separar una de estas tres clases de las otras dos
#   Esto es hacer una estrategia "uno contra todos", clasificacion en 2 clases
#  nothing para separar las 3 clases, clasificación en más de dos clases
# ClaseSeparar = "setosa";
# ClaseSeparar = "virginica";
# ClaseSeparar = "versicolor";
ClaseSeparar = nothing;

################################################################################################
################################################################################################

# Cargamos la base de datos de iris
iris = readdlm("audio.txt");

# Las entradas van a ser el primer y segundo atributo
entradas = iris[:,1:2];
# Cambiamos el tipo de la variable a un array de valores reales
entradas = convert(Array{Float64}, entradas);
# Ademas, en una RNA los patrones estan en columnas, no en filas, asi que trasponemos esa matriz
entradas = Array(entradas');

# La salida deseada va a ser la ultima columna
salidasDeseadas = iris[:,end];
# Cambiamos el tipo de la variable a un array de strings
salidasDeseadas = convert(Array{String}, salidasDeseadas);
# Que clases posibles tenemos
clasesPosibles = unique(salidasDeseadas);

# Si queremos separar las 3 clases
if (ClaseSeparar==nothing)
    # Para separar las 3 clases, es necesario convertir los salidas deseadas en valores booleanos: hacer un "one hot encoding"
    #  Es decir, una matriz donde cada patron tiene varias salidas deseadas: una por clase
    #  La salida deseada de un patron es 1 para la clase a la que pertenece, y 0 para el resto
    # La funcion onehotbatch ya devuelve los patrones traspuestos (en columnas, no en filas)
    salidasDeseadas = Array(onehotbatch(salidasDeseadas, clasesPosibles));
else
    # Se quiere separar solo una clase
    @assert (isa(ClaseSeparar,String)) "La clase a separar no es un String";
    @assert (any(ClaseSeparar.==clasesPosibles)) "La clase a separar no es una de las clases"
    # Para separar una clase, se toma un valor booleano para cada patron: true para los que pertenecen a esa clase, false para los que no
    #  Se traspone porque para una RNA los patrones estan en columnas, no en filas
    salidasDeseadas = Array((salidasDeseadas.==ClaseSeparar)');
end;

# Cuantos patrones, entradas y salidas tenemos
# Importante recordar que en una RNA los patrones estan en las columnas, no en las filas
numPatrones = size(entradas, 2);
numEntradasRNA = size(entradas,1);
numSalidasRNA = size(salidasDeseadas,1);

# Nos aseguramos de que las matrices de entradas y salidas deseadas tengan las dimensiones correctas
@assert (size(entradas,2)==size(salidasDeseadas,2));
# Si vamos a separar dos clases, solamente necesitamos una salida
#  Por tanto, nos aseguramos de que no haya 2 salidas
@assert (numSalidasRNA!=2);


# Vamos a normalizar las entradas entre un maximo y un minimo
# En este caso, al trabajar con redes de neuronas, cada patron esta en una columna
valoresNormalizacion = normalizarMaxMin!(entradas; patronesEnFilas=false);

# Dado que es un problema de clasificacion, no hay que normalizar la salida
# En un problema de regresion si que habria que normalizarla

# Creamos un par de vectores vacios donde vamos a ir guardando los resultados de cada ejecucion
precisionesEntrenamiento = Array{Float64,1}();
precisionesValidacion = Array{Float64,1}();
precisionesTest = Array{Float64,1}();


# Hacemos varias ejecuciones para promediar los resultados
for numEjecucion in 1:numEjecuciones

    println("Ejecucion ", numEjecucion);

    # Para evitar sobreentrenamiento en la RNA, usamos la tecnica de parada temprana
    #  Para esto se necesita un conjunto de validacion
    # Por tanto, separamos los patrones en conjuntos de entrenamiento, validacion y test: creamos indices para cada uno
    # Esto es hacer un hold out
    (indicesPatronesEntrenamiento, indicesPatronesValidacion, indicesPatronesTest) = holdOut(numPatrones, PorcentajeValidacion, PorcentajeTest);

    # Que funcion de transferencia se va a usar en las capas ocultas
    # funcionTransferenciaCapasOcultas = relu;
    funcionTransferenciaCapasOcultas = σ; # sigmoidal

    # Definimos la RNA

    # Si la RNA tiene solo una salida, se aplica la funcion sigmoidal en la capa de salida
    # Si tiene mas de una salida, se aplica una funcion softmax de tal forma que
    #  cada neurona nos devuelva la probabilidad de pertenencia a esa clase,
    #  y la suma de probabilidades sea 1
    if (length(ArquitecturaRNA)==1)
        # Una capa oculta
        if (numSalidasRNA==1)
            # Una unica salida (2 clases)
            modelo = Chain(
                Dense(numEntradasRNA,     ArquitecturaRNA[1], funcionTransferenciaCapasOcultas),
                Dense(ArquitecturaRNA[1], numSalidasRNA,      σ                               ), );
        else
            # Mas de dos salidas/clases
            modelo = Chain(
                Dense(numEntradasRNA,     ArquitecturaRNA[1], funcionTransferenciaCapasOcultas),
                Dense(ArquitecturaRNA[1], numSalidasRNA,      identity),
                softmax, );
        end;
    elseif (length(ArquitecturaRNA)==2)
        # Dos capas ocultas
        if (numSalidasRNA==1)
            # Una unica salida (2 clases)
            modelo = Chain(
                Dense(numEntradasRNA,     ArquitecturaRNA[1], funcionTransferenciaCapasOcultas),
                Dense(ArquitecturaRNA[1], ArquitecturaRNA[2], funcionTransferenciaCapasOcultas),
                Dense(ArquitecturaRNA[2], numSalidasRNA,      σ                               ), );
        else
            # Mas de dos salidas/clases
            modelo = Chain(
                Dense(numEntradasRNA,     ArquitecturaRNA[1], funcionTransferenciaCapasOcultas),
                Dense(ArquitecturaRNA[1], ArquitecturaRNA[2], funcionTransferenciaCapasOcultas),
                Dense(ArquitecturaRNA[2], numSalidasRNA,      identity                        ),
                softmax, );
        end;
    else
        error("Para redes MLP, no hacer mas de dos capas ocultas")
    end;

    # Esta forma de crear la RNA es analoga a la de arriba. De esta forma, se crea una red vacia y se le van añadiendo capas:

    # # Crear la RNA vacia
    # modelo = Chain( );
    # numSalidasCapaAnterior = numEntradasRNA;
    # # Para cada capa oculta
    # for numNeuronasCapa in ArquitecturaRNA
    #     # Añadimos la capa oculta a la RNA
    #     modelo = Chain( modelo..., Dense(numSalidasCapaAnterior, numNeuronasCapa, funcionTransferenciaCapasOcultas) );
    #     numSalidasCapaAnterior = numNeuronasCapa;
    # end;
    # # Ponemos la capa de salida
    # if (numSalidasRNA==1)
    #     modelo = Chain( modelo..., Dense(ArquitecturaRNA[end], numSalidasRNA, σ) );
    # else
    #     modelo = Chain( modelo..., Dense(ArquitecturaRNA[end], numSalidasRNA, identity), softmax );
    # end;

    # Definimos la funcion de error o perdida (loss) de la RNA
    function loss(x, y)
        # Es distinta si es un problema de dos clases, o de mas de dos clases
        if (size(y,1)>1)
            # Problema de mas de dos clases
            #  modelo es la RNA devuelta por el proceso de entrenamiento
            #   se puede usar como funcion: si se pasan entradas, devuelve la salida de la RNA
            #  Se calcula la funcion crossentropy entre las salidas de la RNA y las salidas deseadas
            return crossentropy(modelo(x), y)
        else
            # Problema de dos clases
            #  Se calcula la funcion crossentropy binaria entre las salidas de la RNA y las salidas deseadas
            return mean(binarycrossentropy.(modelo(x), y))
        end;
    end;
    # Esta definicion de la funcion de loss se puede hacer con la siguiente linea:
    loss(x, y) = (size(y,1)>1) ? crossentropy(modelo(x), y) : mean(binarycrossentropy.(modelo(x), y));




    ################################################################################################
    ################################################################################################
    # Entrenamos la RNA hasta que se cumpla el criterio de parada
    criterioFin = false;
    numCiclo = 0; mejorLossValidacion = Inf; numCiclosSinMejorarValidacion = 0; mejorModelo = nothing;
    while (!criterioFin)

        # Entrenamos la RNA un ciclo
        Flux.train!(loss, params(modelo), [(entradas[:,indicesPatronesEntrenamiento], salidasDeseadas[:,indicesPatronesEntrenamiento])], ADAM(0.01));

        numCiclo += 1;

        # Aplicamos el conjunto de validacion a la RNA
        if (PorcentajeValidacion>0)
            lossValidacion = loss(entradas[:,indicesPatronesValidacion], salidasDeseadas[:,indicesPatronesValidacion]);
            if (lossValidacion<mejorLossValidacion)
                mejorLossValidacion = lossValidacion;
                mejorModelo = deepcopy(modelo);
                numCiclosSinMejorarValidacion = 0;
            else
                numCiclosSinMejorarValidacion += 1;
            end;
        end;

        # Criterios para ver si hay que dejar de entrenar
        if (numCiclo>=NumMaxCiclosEntrenamiento)
            criterioFin = true;
        end;
        if (numCiclosSinMejorarValidacion>NumMaxCiclosSinMejorarValidacion)
            criterioFin = true;
        end;

    end;
    # Si estabamos utilizando conjunto de validacion, la rna a devolver sera la que mejor resultado haya tenido en el conjunto de validacion
    if (PorcentajeValidacion>0)
        modelo = mejorModelo;
    end;
    # Fin del entrenamiento de la RNA
    ################################################################################################
    ################################################################################################



    println("   RNA entrenada durante ", numCiclo, " ciclos");

    # Definimos la precision: Se define la siguiente funcion
    function precision(x, y)
        # Es distinta si es un problema de dos clases, o de mas de dos clases
        if (size(y,1)==1)
            # Problema de dos clases
            #  modelo es la RNA devuelta por el proceso de entrenamiento
            #   se puede usar como funcion: si se pasan entradas, devuelve la salida de la RNA
            #  Se pasa un umbral a la salida de la RNA en 0.5 y se compara con las salidas deseadas (valores booleanos)
            return mean((modelo(x).>=0.5) .== y)
        else
            # Problema de mas de dos clases
            #  onecold hace lo contrario de onehot: convierte booleanos a valores enteros: el indice de la clase
            #   se le pueden pasar valores flotantes, toma como indice aquella clase que tiene un mayor valor
            return mean(onecold(modelo(x)) .== onecold(y))
        end;
    end;
    # Esta linea es igual a la definicion de la funcion anterior:
    precision(x, y) = (size(y,1)==1) ? mean((modelo(x).>=0.5) .== y) : mean(onecold(modelo(x)) .== onecold(y));


    # Calculamos los valores de precision obtenidos en los conjuntos de entrenamiento, validacion y test, los mostramos y los almacenamos

    precisionEntrenamiento = 100*precision(entradas[:,indicesPatronesEntrenamiento], salidasDeseadas[:,indicesPatronesEntrenamiento]);
    println("   Precision en el conjunto de entrenamiento: $precisionEntrenamiento %");
    push!(precisionesEntrenamiento, precisionEntrenamiento);

    if (PorcentajeValidacion>0)
        precisionValidacion    = 100*precision(entradas[:,indicesPatronesValidacion],    salidasDeseadas[:,indicesPatronesValidacion]);
        println("   Precision en el conjunto de validacion: $precisionValidacion %");
        push!(precisionesValidacion,    precisionValidacion);
    end;

    precisionTest          = 100*precision(entradas[:,indicesPatronesTest],          salidasDeseadas[:,indicesPatronesTest]);
    println("   Precision en el conjunto de test: $precisionTest %");
    push!(precisionesTest,          precisionTest);

end;


println("Resultados en promedio:");
println("   Entrenamiento: ", mean(precisionesEntrenamiento), " %, desviacion tipica: ", std(precisionesEntrenamiento));
if (PorcentajeValidacion>0)
    println("   Validacion:    ", mean(precisionesValidacion),    " %, desviacion tipica: ", std(precisionesValidacion));
end;
println("   Test:          ", mean(precisionesTest),          " %, desviacion tipica: ", std(precisionesTest));
