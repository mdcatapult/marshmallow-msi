
#!/usr/bin/env Rscript
# if (!requireNamespace("BiocManager", quietly=TRUE))
#     install.packages("BiocManager",repos = "https://cloud.r-project.org/")

# BiocManager::install("Cardinal")
library('Cardinal')

args = commandArgs(trailingOnly = TRUE)

#ncores = as.integer(readline(prompt='Enter number of cores: '))
#folder = readline(prompt='Enter imzml folder path: ')
#ppmethod = readline(prompt = 'Enter peak-picking methods cwt/filter: ')
#snr  = as.integer(readline(prompt = 'Enter SNR: '))
#intra_alignment_tolerance = as.integer(readline(prompt='Enter pixel m/z alignment tolerance:'))

ncores = as.integer(args[1])
folder = args[2]
ppmethod = args[3]
snr  = as.integer(args[4])
intra_alignment_tolerance = as.integer(args[5])
# freq = as.numeric(args[6])


dir.create(file.path(folder, 'peaks'), showWarnings = FALSE)
out_folder = file.path(folder, 'peaks/')

setCardinalBPPARAM(MulticoreParam(ncores, progressbar = TRUE))

files = list.files(path=folder, pattern='.imzML', full.names = TRUE)
print(files)
nid = length(strsplit(folder, split = "/")[[1]])

#check if baselining is necessary
data = readMSIData(files[1])
msa_pre = process(data)
plot(msa_pre)



for (fileI in files)
  {
    print(file.info(fileI)$size)
    
    
    id = strsplit(fileI, split = "/")[[1]][nid+1]
    print(id)
    
    data = readMSIData(fileI)
    
    msa_pre = process(data)
    msa_pre = smooth(msa_pre, method="gaussian")
    msa_pre = reduceBaseline(msa_pre, method="locmin")
    # if (fileI == files[1])
    #   { 
    #     plot(msa_pre)
    #     baseline  = readline(prompt = 'Reduce baseline? TRUE/FALSE: ')
    # }
    # 
    # if (baseline)
    #   {
    #     msa_pre = reduceBaseline(msa_pre, method="locmin")
    #   }
    # 
    
    start = Sys.time()
    mse = peakProcess(msa_pre, method =  ppmethod, SNR = snr, type = 'height', tolerance = intra_alignment_tolerance, units = 'ppm')
    time = Sys.time() - start
    print(time)
    
    #imzfile = paste('/Volumes/T7_Shield/MDCP-0393-Skou/re-test-imzml/height/',id, sep='')
    #writeMSIData(mse, file=imzfile)
    
    mse_filt = subsetFeatures(mse, freq > 0.05)
    # mse_filt = subsetFeatures(mse_filt, mz<1000)
    
    imzfile_filt = paste(out_folder,id, sep='')
    writeMSIData(mse_filt, file=imzfile_filt)
    
    #rm(list=ls())
    gc()
}

print(out_folder)