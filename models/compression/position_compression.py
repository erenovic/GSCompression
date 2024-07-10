import subprocess
import os
import torch

rootdir = os.path.join(
    os.path.abspath(os.path.curdir), "models", "compression", 
    "mpeg-pcc-tmc13", "build", "tmc3"
)


def gpcc_encode(filedir, bin_dir, show=False):
    """Compress point cloud losslessly using MPEG G-PCCv12. 
    You can download and install TMC13 from 
    https://github.com/MPEGGroup/mpeg-pcc-tmc13
    """
    subp=subprocess.Popen(rootdir+'/tmc3'+ 
                            ' --mode=0' + 
                            ' --positionQuantizationScale=1' + 
                            ' --trisoupNodeSizeLog2=0' + 
                            ' --mergeDuplicatedPoints=0' +
                            ' --neighbourAvailBoundaryLog2=8' + 
                            ' --intra_pred_max_node_size_log2=6' + 
                            ' --inferredDirectCodingMode=0' + 
                            ' --maxNumQtBtBeforeOt=4' +
                            ' --minQtbtSizeLog2=0' +
                            ' --partitionOctreeDepth=15'
                            ' --uncompressedDataPath='+filedir + 
                            ' --compressedStreamPath='+bin_dir, 
                            shell=True, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    while c:
        if show: print(c)
        c=subp.stdout.readline()
    
    return 


def gpcc_decode(bin_dir, rec_dir, show=False):
    subp=subprocess.Popen(rootdir+'/tmc3'+ 
                            ' --mode=1'+ 
                            ' --compressedStreamPath='+bin_dir+ 
                            ' --reconstructedDataPath='+rec_dir+
                            ' --outputBinaryPly=0'
                          ,
                            shell=True, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    while c:
        if show: print(c)      
        c=subp.stdout.readline()
    
    return


def get_sorting(rec_tensor: torch.Tensor, enc_tensor: torch.Tensor, B=1000):
    # Compute pairwise squared distances
    rec_tensor = rec_tensor.cuda()
    enc_tensor = enc_tensor.cuda()

    sort_indices = torch.empty(0).cuda()

    with torch.no_grad():
        for idx in range((rec_tensor.shape[0] // B) + 1): 
            diff = rec_tensor[B*idx:B*(idx+1), None, :] - enc_tensor[None, :, :]
            distances = torch.sum(diff ** 2, dim=2)
            min_distances, min_indices = torch.min(distances, dim=1)
            sort_indices = torch.cat((sort_indices, min_indices))

        sorted_indices = sort_indices.cpu().long()
    
    return sorted_indices