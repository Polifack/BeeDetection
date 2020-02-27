using FFTW
using WAV
using Statistics
using PlotlyJS
using DelimitedFiles

#Inputs:
#   ampSignal -> Señal en amplitud
#   frameTime -> Duracion en segundos del marco de la señal
#   sampFreq -> Frecuencia de sampleo de la señal
#Output: 
#   freqSign -> Señal en frecuencia
function fourier(ampSignal, frameTime, sampFreq) 
    freqSignal = abs.(fft(ampSignal))
    chanLength = length(ampSignal)
    nFrames = trunc(Int,frameTime * sampFreq)
    #Comprobacion de valores
    if (iseven(nFrames))
        @assert(mean(abs.(freqSignal[2:Int(nFrames/2)] .- freqSignal[end:-1:(Int(nFrames/2)+2)]))<1e-8)
        freqSignal = freqSignal[1:(Int(nFrames/2)+1)]
    else
        @assert(mean(abs.(freqSignal[2:Int((nFrames+1)/2)] .- freqSignal[end:-1:(Int((nFrames-1)/2)+2)]))<1e-8)
        freqSignal = freqSignal[1:(Int((nFrames+1)/2))]
    end
    return freqSignal
end

#Inputs:
#   fileName -> nombre del archivo
#Output:
#   devuelve un array de arrays que contiene los datos del archivo de datos
#   el array tiene n (numero de muestras del archivo) elementos
#   cada uno de estos elementos tiene 3 elementos
#       [1] -> tiempo inicio
#       [2] -> tiempo fin
#       [3] -> 'bee' o 'nobee'
function processLabfile(fileName)
    println("[*] Procesando el archivo lab: ",fileName)
    samplesFile = readdlm(fileName, header=true)

    #Procesamos los valores de los archivos
    sampleDataRaw = samplesFile[1]
    sampleName = samplesFile[2]

    #Procesamos los datos
    startTimes = sampleDataRaw[1:end-1,1];
    endTimes = sampleDataRaw[1:end-1,2];
    hasBee = sampleDataRaw[1:end-1,3];  

    temp = []
    for x = 1:length(hasBee)
        element = [startTimes[x], endTimes[x], hasBee[x]]
        push!(temp, element)
    end
    println("[*] Procesado del archivo lab completado")
    return temp
end

#Inputs:
#   labData -> resultado de processLabFile, array de 3 elementos donde
#       [1] -> tiempo inicio
#       [2] -> tiempo fin
#       [3] -> tiempo has_bee
#   audioChannel -> array de doubles que son el resultado del wavread
#   sampleFrequency -> frecuencia de muestreo
#Output:
#   devuelve un array de elementos compuestos por
#       [1] -> trim de doubles que representan el sonido en amplitud, es un trim del audioChannel
#       [2] -> 1.0 o 0.0 que indica si en dicho segmento hay abeja o no 
function splitAudioWithLabfile(labData, audioChannel, sampleFrequency)
    println("[*] Separando el archivo de audio con el archivo lab")
    sound = []

    #Calculamos el numero de muestras del audio y su duracion en segundos
    nsamples = length(audioChannel)/sampleFrequency
    duracion = nsamples / sampleFrequency
    
    for x=1:length(labData)
        #Formalizamos los datos del array
        start_time = labData[x][1]
        end_time = labData[x][2]
        length = end_time - start_time
        has_bee = false
        if (labData[x][3]=="bee") 
            has_bee = true
        end

        #Separamos el audio
        prop_start = start_time/duracion
        prop_end = end_time/duracion

        sample_start = trunc(Int, prop_start * nsamples + 1)
        sample_end = trunc(Int, prop_end * nsamples)
        
        #Pillamos los samples del audio en las frecuencias deseadas
        audio_trim = []
        for i=sample_start:sample_end
            push!(audio_trim, audioChannel[i])
        end

        #Creamos un par del tramo de audio deseado y el bool que indica si hay abeja o no
        element = [audio_trim, has_bee]
        
        #Añadimos el elemento al array general 
        push!(sound, element)
    end
    println("[*] Separacion del archivo de audio completada")
    return sound
end

#Inputs:
#   windowLength -> tamaño de la ventana de audio que vamos a procesar de cada vez
#   offsetLength -> tamaño del desplazamiento que vamos a procesar de cada vez
#   audioChannel -> pista de audio que vamos a procesar
#   sampleFrequency -> frecuencia de muestreo, necesaria para calcular nSamples
#   hasBee -> indicador de si en el audio audioChannel hay abjea
#Output
#   devuelve un array de elementos compuestos por
#       [1] -> media de la fft del audio en la ventana indicada
#       [2] -> desv tipica de la fft del audio en la ventana indicada
#       [3] -> 1.0 o 0.0 indicando si hay abeja en el audio o no
function parseAudioSample(windowLength, offsetLength, audioChannel, sampleFrequency, hasBee)
    println("[*] Procesando segmento de audio con tamaño de ventana = ", windowLength)
    windowSamples = trunc(Int, windowLength * sampleFrequency)
    audioSamples = length(audioChannel)
    
    #parametros que nos sirven para ver donde empieza y acaba la ventana
    begin_window = 0
    end_window = (begin_window+windowSamples)

    #array de los resultados de procesar el audio
    result = []

    #repetimos hasta que la ultima ventana no se pueda procesar entera, esto es, descartamos lo que sobra
    while (end_window < audioSamples)
        #Array temporal que contiene el audio de la ventana
        audioWindowAmp = []
        #Obtenemos las samples de audio de la ventana
        for x=(begin_window+1):(end_window)
            push!(audioWindowAmp, audioChannel[x])
        end
        #Movemos las ventanas
        begin_window+=trunc(Int, (offsetLength*sampleFrequency))
        end_window+=trunc(Int, (offsetLength*sampleFrequency))

        #Convertimos el array a Floats
        audioWindowAmp = convert(Array{Float64,1}, audioWindowAmp)
        audioWindowFreq = fourier(audioWindowAmp, windowLength, sampleFrequency)

        media = mean(audioWindowFreq[1:length(audioWindowFreq)])
        desviacion = std(audioWindowFreq[1:length(audioWindowFreq)]);

        #Elemento que contiene la media, desviacion y si hay abeja o no
        element = [media, desviacion, hasBee]
        push!(result, element)
    end
    println("[*] Procesado del segmento de audio completado")
    return result
end

#Inputs:
#   soundData -> array de los datos del archivo de audio procesado
#       [1] -> array de segmentos con las frecuencias del audio
#       [2] -> array de segmentos con un 1.0 o 0.0 que indica si existe abeja
#   windowSize -> tamaño de la ventana que queremos procesar
#   windowOffset -> tamaño del desplazamiento que vamos a procesar de cada vez
#   frequency -> frecuencia de muestreo
#Output
#   devuelve un array de elementos de TODO el wav compuestos por
#       [1] -> media de la fft del audio en la ventana indicada
#       [2] -> desv tipica de la fft del audio en la ventana indicada
#       [3] -> 1.0 o 0.0 indicando si hay abeja en el audio o no
function process_wav(soundData, windowSize, windowOffset, frequency)
    println("[*] Procesando archivo wav")
    parsed_audio = []
    for x=(1:length(soundData))
        audio = soundData[x][1]
        has_bee = soundData[x][2]

        parsed_frame = parseAudioSample(windowSize, windowOffset, audio, frequency, has_bee)
        push!(parsed_audio, parsed_frame)
    end

    result = []
    for x=(1:length(parsed_audio))
        subarray = parsed_audio[x]
        for y=(1:length(subarray))
            element = subarray[y]
            push!(result, element)
        end
    end
    return result
    println("[*] Procesado del archivo wav completado")
    return (parsed_audio)
end


#Inputs:
#   audioPath -> path al archivo del audio de la muestra
#   labPath -> path al archivo .lab relacionado con el audio
#   fileName -> path al archivo.txt que vamos a escribir los resultados
#   windowSize -> tamaño de la ventana que queremos procesar
#   windowOffset -> tamaño del desplazamiento que vamos a procesar de cada vez
function process_sample(audioPath, labPath, fileName, windowSize, windowOffset)
    println("[*] Iniciando procesado de ",audioPath," con archivo de datos ",labPath)
    #Leemos el labfile y el audio
    labData = processLabfile(labPath)
    sound, freq = wavread(audioPath, format="double")
    #Separamos los canales
    audioChannel = sound[:,1]
    #Procesamos el sonido con la labfile
    sound  = splitAudioWithLabfile(labData, audioChannel, freq)
    #Parseamos el audio para obtener los parametros y la salida deseada
    parsedAudio = process_wav(sound, windowSize, windowOffset, freq)   
    
    println("[*] Escribiendo resultados en ", fileName)
    #Escribimos en el archivo los datos procesados
    f = open(fileName, "a")
    for i in eachindex(parsedAudio)
        print(f, round.(parsedAudio[i][1]; digits=3)," ")
        print(f, round.(parsedAudio[i][1]; digits=3)," ")
        
        has_bee = (parsedAudio[i][3]==1.0)
        println(f, has_bee)
    end
    close(f)
    println("[*] Procesado completado")
end


#Usage julia ./procesado.jl 'song.wav' 'data.lab'
args = ARGS
cancion = args[1]
lab = args[2]
fileName = "audio.txt"
#Borramos el archivo si existe
rm(fileName)
process_sample(cancion, lab,"audio.txt",3, 1)

