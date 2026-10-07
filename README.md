# SQUID 3D shape generation

Generates up to ten molecules that fill the three-dimensional shape of a reference compound while letting the chemistry change, the central move of ligand-based scaffold hopping. SQUID, from Adams and Coley, encodes shape with an equivariant point cloud network, variationally encodes chemical identity and assembles molecules fragment by fragment while scoring rotatable bonds, after training on drug-like molecules from MOSES. Ersilia seeds every sampling attempt, so a given input reproduces, and inputs whose ring systems fall outside the fixed fragment vocabulary return nothing.

This model was incorporated on 2024-05-01.Last packaged on 2026-09-28.

## Information
### Identifiers
- **Ersilia Identifier:** `eos8vud`
- **Slug:** `squid`

### Domain
- **Task:** `Sampling`
- **Subtask:** `Generation`
- **Biomedical Area:** `Any`
- **Target Organism:** `Any`
- **Tags:** `Compound generation`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `10`
- **Output Consistency:** `Variable`
- **Interpretation:** Up to ten molecules generated to match the three-dimensional shape of the input, with seeded sampling.

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| smi_0 | string |  | This input index was calculated using the pretrained SQUID model |
| smi_1 | string |  | This input index was calculated using the pretrained SQUID model |
| smi_2 | string |  | This input index was calculated using the pretrained SQUID model |
| smi_3 | string |  | This input index was calculated using the pretrained SQUID model |
| smi_4 | string |  | This input index was calculated using the pretrained SQUID model |
| smi_5 | string |  | This input index was calculated using the pretrained SQUID model |
| smi_6 | string |  | This input index was calculated using the pretrained SQUID model |
| smi_7 | string |  | This input index was calculated using the pretrained SQUID model |
| smi_8 | string |  | This input index was calculated using the pretrained SQUID model |
| smi_9 | string |  | This input index was calculated using the pretrained SQUID model |


### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`
- **DockerHub**: [https://hub.docker.com/r/ersiliaos/eos8vud](https://hub.docker.com/r/ersiliaos/eos8vud)
- **Docker Architecture:** `AMD64`, `ARM64`
- **S3 Storage**: [https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos8vud.zip](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos8vud.zip)

### Resource Consumption
- **Model Size (Mb):** `371`
- **Environment Size (Mb):** `2608`
- **Image Size (Mb):** `3052.22`

**Computational Performance (seconds):**
- 10 inputs: `31.17`
- 100 inputs: `1512.13`
- 10000 inputs: `-1`

### References
- **Source Code**: [https://github.com/keiradams/SQUID](https://github.com/keiradams/SQUID)
- **Publication**: [https://doi.org/10.48550/arXiv.2210.04893](https://doi.org/10.48550/arXiv.2210.04893)
- **Publication Type:** `Preprint`
- **Publication Year:** `2023`
- **Ersilia Contributor:** [miquelduranfrigola](https://github.com/miquelduranfrigola)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [MIT](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos8vud
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos8vud
# generate an example file
ersilia example -n 3 -f my_input.csv
# run the model
ersilia run -i my_input.csv -o my_output.csv
# close the model
ersilia close
```

## About Ersilia
The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization fueling sustainable research in the Global South.
Please [cite](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff) the Ersilia Model Hub if you've found this model to be useful. Always [let us know](https://github.com/ersilia-os/ersilia/issues) if you experience any issues while trying to run it.
If you want to contribute to our mission, consider [donating](https://www.ersilia.io/donate) to Ersilia!
