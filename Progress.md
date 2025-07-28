### Datasets:
We take time duration of 2000-2019 for both datasets.
<br>
Datasets are in folder `pmelData` ([link](https://www.pmel.noaa.gov/tao/drupal/disdel/)) and `uknet` ([link](https://www.metoffice.gov.uk/hadobs/en4/download-en4-2-2.html)). 
    - For pmel dataset, 6S10W is chosen, since it has most data coverage.
    - For uknet dataset, we choose same location as pmel dataset. 'Levitus et al. (2009) corrections' taken.

### **Week 1**  [Folder: Code_1]
We first predict the density values using the standard non-linear EOS for seawater (in gsw library) for both the datasets. 
- **Observations:**
    - The pmel dataset had density values available. So comparing the predicted values, we see very small differences (order of 10^-3).
- For the uknet dataset, we took depths <200m. 

<br>

Now we calculate the MLD using threshold of 0.125 kg/m^3 (accurate) and 0.2 kg/m^3 (relaxed).
- See file `Code_1/ukmet_code/mixDepth.ipynb` for details.
