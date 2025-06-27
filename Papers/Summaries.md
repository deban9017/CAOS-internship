1. **DD-PINN (DT perspective) paper:** 
    <br>
    - Data-PINN better than no data PINN (only physics) - minima are steeper for DD-PINN.
    - *Sampling*: Random sampling works best for DD-PINN. For ND-PINN (ND: No data) we can use various types of sampling like gradient based, etc. Eg. vorticity based, residue based sampling. (Basically train the model somewhat. Then identify when/where the model can't predict so accurately, sample more training points from that region). Also can define probability distribution for sampling.
    - For those parameters due to which high frequency changes occur (eg. high Reynolds number), we can use multiscale sampling. Eg. sample from 20x20 grid for low Reynolds number and 160x160 grid for high Reynolds number.



