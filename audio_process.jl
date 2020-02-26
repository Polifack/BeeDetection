using FFTW
using WAV
using Statistics
using PlotlyJS
using DelimitedFiles


args = ARGS

#Usage julia ./procesado.jl 'song.wav' 'data.lab'

cancion = args[1]
lab = args[2]

println(cancion)
println(lab)


#Inputs:
#   ampSignal -> Señal en amplitud
#   frameTime -> Duracion en segundos del marco de la señal
#   sampFreq -> Frecuencia de sampleo de la señal
#Output: 
#   freqSign -> Señal en frecuencia
function fourier(ampSignal, frameTime, sampFreq) 
    freqSignal = abs.(fft(ampSignal))
    chanLength = length(ampSignal)
    nFrames = frameTime * sampFreq
    #Comprobacion de valores
    if (iseven(nsamples))
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
    samplesFile = readdlm(fileName, header=true)

    #Procesamos los valores de los archivos
    sampleDataRaw = samplesFile[1]
    sampleName = samplesFile[2]

    #Procesamos los datos
    startTimes = sampleDataRaw[1:end-1,1];
    endTimes = sampleDataRaw[1:end-1,2];
    hasBee = sampleDataRaw[1:end-1,3];  

    createArray(startTimes, endTimes, hasBee)
end
function createArray(startTimes, endTimes, classes)
    temp = []
    for x = 1:length(classes)
        element = [startTimes[x], endTimes[x], classes[x]]
        push!(temp, element)
    end

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
#       [2] -> bool que indica si en dicho segmento hay abeja o no 
function splitAudioWithLabfile(labData, audioChannel, sampleFrequency)
    sound = []

    #Calculamos el numero de muestras del audio y su duracion en segundos
    nsamples = length(audioChannel)/sampleFrequency
    duracion = nsamples / sampleFrequency
    
    for x=1:length(labdata)
        #Formalizamos los datos del array
        start_time = labdata[x][1]
        end_time = labdata[x][2]
        length = end_time - start_time
        has_bee = false
        if (labdata[x][3]=="bee") 
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
            push!(audio_trim, audio_channel[i])
        end

        #Creamos un par del tramo de audio deseado y el bool que indica si hay abeja o no
        element = [audio_trim, has_bee]
        
        #Añadimos el elemento al array general 
        push!(sound, element)
    end

    return sound
end

###############
#PROCESAR AUDIO
###############

    #Leemos el audio
    sound, freq = wavread(cancion, format="double")
    #separamos los canales
    audio_channel = sound[:,1]

#################
#PROCESAR LABFILE
#################
    
    labdata = processLabfile(lab)

#################
#PROCESAR DATOS
#################
    
    #Separamos el audio segun haya abeja o no
    sound  = splitAudioWithLabfile(labdata, audio_channel, freq)
    
    #Printeamos el array de sonidos
    for x=1:length(sound)
        print(sound[x])
    end

    #Parseamos cada uno de los segmentos segun un tiempo n

    




