# AI Translation Validation for Community Contributions in Open Source Projects (Ubuntu Desktop i18n).
Master thesis conducted at Canonical, a leading company in the development and delivery of open-source software. Canonical is renowned for creating Ubuntu, one of the most widely used Linux distributions worldwide.

### A project of interest for Canonical
Open-source projects, such as Ubuntu, rely on community contributions. The source code of the project is publicly available, allowing any developer to participate in its development. This includes coding new features, fixing bugs, but also suggesting or updating translations to make the software accessible to a broader range of users.

Community-based translations offer the advantage of providing accurate, human-friendly translations, which AI-based translation tools might fail to achieve due to the lack of appropriate vocabulary, especially for small strings that lack context in an operating system. However, before integrating these changes into the final version of the software, a preliminary check must be performed to assess the quality of the contribution. It is crucial to ensure that the new translations are relevant, precise, and free from malicious content.

The Ubuntu 23.10 release was targeted by a harmful actor who contributed hateful speech to the Ukrainian version of the installer. Thus, Canonical aims to maintain human involvement in the translation process while using AI tools to enhance robustness.

### RESAERCH QUESTION
How to evaluate the quality of a translation and/or detect possible mistranslations in a document ?

## Subdivision of the project

### 1. Data collection
The process of data collection consists in gathering translations from several open source operatin systems :
- OPENSuse,
- Debian,
- Ubuntu.

We made sure that data gathered from different sources are :
- Formatted the same way and stored grouped by language,
- Easy to update, 
- Stored on HuggingFace : https://huggingface.co/datasets/RomainDarous/open-source-os-translations.

The code of data collection is available in the folder ``1_data_collection``

### 2. Data processing
Processing the data consists in making the collected data ready for training models on it. It includes :
- Data cleaning,
- Splitting into training/validation/test sets,
The code of data processing is available in the folder ``2_data_processing``

### 3. Model design
Backed up with background research to be up-to-date regarding state-of-the-art techniques in the fields of anomaly detection, translation models, translation quality metrics and sentiment analysis, a model will be built to tackle the problem as efficiently as possible.

The code of model design is available in the folder ``3_model``