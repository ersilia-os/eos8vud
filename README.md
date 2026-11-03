# SQUID 3D shape generation

Generates up to ten molecules conditioned on the 3D shape of a reference compound, for ligand-based scaffold hopping. SQUID, from Adams and Coley, encodes the shape of a 3D conformer with an equivariant point cloud network and grows each molecule from a seed fragment of the input, using a fixed vocabulary of 100 ring templates from MOSES. Sampling is stochastic; the input, its stereoisomers, multi-component structures and repeats are removed. Most growth attempts fail, so rows are often short. Inputs with other ring systems, and empty, unparsable or multi-component inputs, return empty rows.

This model was incorporated on 2024-05-01.
Last packaged on 2026-09-28.

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
- **Interpretation:** Up to ten molecules generated conditioned on the three-dimensional shape of the input.

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| smi_0 | string |  | Generated molecule index 0 conditioned on the 3D shape of the input (empty if fewer than 10 were found) |
| smi_1 | string |  | Generated molecule index 1 conditioned on the 3D shape of the input (empty if fewer than 10 were found) |
| smi_2 | string |  | Generated molecule index 2 conditioned on the 3D shape of the input (empty if fewer than 10 were found) |
| smi_3 | string |  | Generated molecule index 3 conditioned on the 3D shape of the input (empty if fewer than 10 were found) |
| smi_4 | string |  | Generated molecule index 4 conditioned on the 3D shape of the input (empty if fewer than 10 were found) |
| smi_5 | string |  | Generated molecule index 5 conditioned on the 3D shape of the input (empty if fewer than 10 were found) |
| smi_6 | string |  | Generated molecule index 6 conditioned on the 3D shape of the input (empty if fewer than 10 were found) |
| smi_7 | string |  | Generated molecule index 7 conditioned on the 3D shape of the input (empty if fewer than 10 were found) |
| smi_8 | string |  | Generated molecule index 8 conditioned on the 3D shape of the input (empty if fewer than 10 were found) |
| smi_9 | string |  | Generated molecule index 9 conditioned on the 3D shape of the input (empty if fewer than 10 were found) |


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
