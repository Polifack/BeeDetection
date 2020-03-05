using DelimitedFiles
using Flux: onehotbatch, onecold, crossentropy, binarycrossentropy
using Flux
using Statistics
using Random

#Hacemos que el julia cuda se calle la boca
ENV["JULIA_CUDA_SILENT"] = true

#Funcion para separar el set de entrenamiento, validacion y testeo
function holdOut(numPatrones::Int, porcentajeValidacion::Float64, porcentajeTest::Float64)
    @assert ((porcentajeValidacion>=0.) & (porcentajeValidacion<=1.));
    @assert ((porcentajeTest>=0.) & (porcentajeTest<=1.));
    @assert ((porcentajeValidacion+porcentajeTest)<=1.);
    indices = randperm(numPatrones);
    numPatronesValidacion = Int(round(numPatrones*porcentajeValidacion));
    numPatronesTest = Int(round(numPatrones*porcentajeTest));
    numPatronesEntrenamiento = numPatrones - numPatronesValidacion - numPatronesTest;
    return (indices[1:numPatronesEntrenamiento], indices[numPatronesEntrenamiento+1:numPatronesEntrenamiento+numPatronesValidacion], indices[numPatronesEntrenamiento+numPatronesValidacion+1:numPatronesEntrenamiento+numPatronesValidacion+numPatronesTest]);
end

# Cargamos la base de datos de iris
bd = readdlm("audio.txt");

# Las entradas son un array que contiene todas las medias y desv tipicas
entradas = bd[:,1:2];
entradas = convert(Array{Float64}, entradas);
entradas = Array(entradas');

#Las salidas es un array que contiene todos los bee/nobee
salidasDeseadas = bd[:,end]
salidasDeseadas = convert(Array{String}, salidasDeseadas);

# Que clases posibles tenemos? Bee y no bee
clasesPosibles = unique(salidasDeseadas);

ClaseSeparar="bee"
@assert (isa(ClaseSeparar,String)) "La clase a separar no es un String";
@assert (any(ClaseSeparar.==clasesPosibles)) "La clase a separar no es una de las clases"

#Creamos un array de bools que indican si los elementos de las salidas son 'salida deseada'
salidasDeseadas = Array((salidasDeseadas.==ClaseSeparar)');

#Obtener parametros del set de entrada
numPatrones = size(entradas, 2);
numEntradasRNA = size(entradas,1);
numSalidasRNA = size(salidasDeseadas,1);

println("numPatrones: ",numPatrones," || numEntradas: ",numEntradasRNA,
    " || numSalidas: ",numSalidasRNA)

#Nos aseguramos de que hay una salida por cada entrada
@assert (size(entradas,2)==size(salidasDeseadas,2));
#Nos aseguramos de que solo haya una salida
@assert (numSalidasRNA!=2);

# Creamos un par de vectores vacios donde vamos a ir guardando los resultados de cada ejecucion
precisionesEntrenamiento = Array{Float64,1}();
precisionesValidacion = Array{Float64,1}();
precisionesTest = Array{Float64,1}();

#Parametros de la RNA
ArquitecturaRNA = [5, 4];
PorcentajeValidacion = 0.2;
PorcentajeTest = 0.2;
NumMaxCiclosEntrenamiento = 1000;
NumMaxCiclosSinMejorarValidacion = 100;
numEjecuciones = 50;

for numEjecucion in 1:numEjecuciones
    println("[*] Separando los conjuntos de TEST, VALIDACION y ENTRENAMIENTO")
    (indicesPatronesEntrenamiento, indicesPatronesValidacion, indicesPatronesTest) = 
        holdOut(numPatrones, PorcentajeValidacion, PorcentajeTest);


    #Definimos las funciones de transferencia enter las capas ocultas y con la capa de salida

    # La funcion de transferencia puede ser σ en caso de que haya una sola salida (nuestro caso)
    # o una función softmax que devuelva un valor de probabilidad de pertenencia a una clase
    funcionTransferenciaCapasOcultas = σ
    funcionTransferenciaCapaSalida = σ
    #La funcion de transferencia con la capa de salida siempre es sigm

    # Creacion de la RNA

    println("[*] Creando RNA con ",length(ArquitecturaRNA)," capas ocultas.")
    #Creacion de una RNA para 1 capa oculta
    if (length(ArquitecturaRNA)==1)
        println("[*] Creando modelo para RNA de 1 capa oculta")
        modelo = Chain(
            Dense(numEntradasRNA,ArquitecturaRNA[1],funcionTransferenciaCapasOcultas),
            Dense(ArquitecturaRNA[1], numSalidasRNA, funcionTransferenciaCapaSalida), );
    #Creacion de una RNA para 2 capas ocultas
    elseif (length(ArquitecturaRNA)==2)
        println("[*] Creando modelo para RNA de 2 capas oculta")
        modelo = Chain(
            Dense(numEntradasRNA,ArquitecturaRNA[1],funcionTransferenciaCapasOcultas),
            Dense(ArquitecturaRNA[1],ArquitecturaRNA[2],funcionTransferenciaCapasOcultas),
            Dense(ArquitecturaRNA[2],numSalidasRNA,funcionTransferenciaCapaSalida), );
    else
        error("Para redes MLP, no hacer mas de dos capas ocultas")
    end;

    #Funcion para calcular el error de la RNA
    loss(x,y) = mean(binarycrossentropy.(modelo(x), y))
    #Funcion para calcular la precision de la RNA
    precision(x,y) = mean((modelo(x).>=0.5) .== y)

    #Iniciamos el entrenamiento de la RNA
    println("[*] Iniciando entrenamiento de la RNA")
    criterioFin = false;
    numCiclo = 0; 
    mejorLossValidacion = Inf; 
    numCiclosSinMejorarValidacion = 0;
    mejorModelo = nothing;
    
    while (!criterioFin)
        Flux.train!(loss, params(modelo), [(entradas[:,indicesPatronesEntrenamiento], 
            salidasDeseadas[:,indicesPatronesEntrenamiento])], ADAM(0.01));
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
        end

        #Dejamos de entrenar cuando llegamos a los ciclos de entrenamiento maximo...
        if (numCiclo>=NumMaxCiclosEntrenamiento)
            criterioFin = true;
        end
        #... o cuando llevamos n ciclos sin obtener mejora
        if (numCiclosSinMejorarValidacion>NumMaxCiclosSinMejorarValidacion)
            criterioFin = true;
        end
    end

    #Devolvemos el mejor modelo de todos los obtenidos
    if (PorcentajeValidacion>0)
        modelo = mejorModelo;
    end;

    println("[*] RNA entrenada durante ", numCiclo, " ciclos");

    precisionEntrenamiento = 100*precision(entradas[:,indicesPatronesEntrenamiento], 
        salidasDeseadas[:,indicesPatronesEntrenamiento]);
    println("[*] Precision en el conjunto de entrenamiento: $precisionEntrenamiento %");
    push!(precisionesEntrenamiento, precisionEntrenamiento);

    if (PorcentajeValidacion>0)
        precisionValidacion = 100*precision(entradas[:,indicesPatronesValidacion],
            salidasDeseadas[:,indicesPatronesValidacion]);
        println("[*] Precision en el conjunto de validacion: $precisionValidacion %");
        push!(precisionesValidacion, precisionValidacion);
    end;

    precisionTest = 100*precision(entradas[:,indicesPatronesTest], 
        salidasDeseadas[:,indicesPatronesTest]);
    println("[*] Precision en el conjunto de test: $precisionTest %");
    push!(precisionesTest, precisionTest);
end

println("*****************************************************")
println("Resultados en promedio:");
println(" -> Entrenamiento: ", mean(precisionesEntrenamiento), " %, desviacion tipica: ", std(precisionesEntrenamiento));
if (PorcentajeValidacion>0)
    println(" -> Validacion: ", mean(precisionesValidacion)," %, desviacion tipica: ", std(precisionesValidacion));
end;
println(" -> Test: ", mean(precisionesTest)," %, desviacion tipica: ", std(precisionesTest));
println("*****************************************************")