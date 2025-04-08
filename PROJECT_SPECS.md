# MLOps Project Specifications

## Project Overview
This project is a reproduction of the [graviraja-mlops-basics](https://github.com/graviraja/mlops-basics) repository, focusing on implementing fundamental MLOps concepts and best practices. The project uses the AG News dataset for text classification, implementing a BERT-based model with PyTorch Lightning.

## Technical Stack

### Core Technologies
- **Python Version**: 3.12.3
- **Deep Learning Framework**: PyTorch Lightning
- **Transformer Model**: BERT (google/bert_uncased_L-2_H-128_A-2)
- **Data Processing**: Hugging Face Datasets, Pandas
- **MLOps Tools**: 
  - DVC (Data Version Control)
  - Hydra (Configuration Management)
  - Docker (Containerization)
  - GitHub Actions (CI/CD)
  - AWS Services (ECR, Lambda)
  - Weights & Biases (Experiment Tracking)

### Key Libraries
- `lightning`: For structured deep learning development
- `transformers`: For BERT model and tokenization
- `datasets`: For data loading and processing
- `pandas`: For data manipulation
- `scikit-learn`: For data splitting and evaluation
- `torch`: For deep learning operations

## Project Structure
```
MLOps-Basics/
├── data.py              # Data loading and preprocessing
├── model.py            # Model architecture and training logic
├── test_notebooks/     # Jupyter notebooks for experimentation
├── requirements.txt    # Project dependencies
└── PROJECT_SPECS.md    # This file
```

## Model Architecture
- **Base Model**: BERT (google/bert_uncased_L-2_H-128_A-2)
- **Task**: Text Classification
- **Input**: Text sequences (max length: 512 tokens)
- **Output**: News category classification
- **Training**: PyTorch Lightning for structured training

## Data Pipeline
1. **Data Loading**: AG News dataset from Hugging Face
2. **Preprocessing**:
   - Text tokenization using BERT tokenizer
   - Sequence padding and truncation
   - Train/Validation split (80/20)
3. **Data Format**: Hugging Face Dataset format with PyTorch tensors

## MLOps Practices Implemented

### 1. Data Version Control (DVC)
- Tracking data versions
- Managing data pipelines
- Reproducible data processing

### 2. Configuration Management (Hydra)
- Centralized configuration
- Environment-specific settings
- Experiment tracking

### 3. Model Monitoring
- Weights & Biases integration
- Performance metrics tracking
- Model versioning

### 4. Containerization
- Docker for consistent environments
- Containerized training and inference
- AWS ECR for container registry

### 5. CI/CD Pipeline
- GitHub Actions for automation
- Automated testing
- Deployment workflows

### 6. Serverless Deployment
- AWS Lambda for inference
- API Gateway integration
- Scalable serving

### 7. Monitoring & Logging
- Kibana dashboards
- CloudWatch integration
- Elasticsearch for log management

## Development Workflow
1. Local development with notebooks
2. Code modularization and testing
3. Containerized training
4. Automated deployment
5. Monitoring and maintenance

## Performance Metrics
- Model accuracy
- Inference latency
- Resource utilization
- API response times

## Future Enhancements
1. Model optimization
2. Additional dataset support
3. Enhanced monitoring
4. A/B testing capabilities
5. Automated retraining pipeline

## Getting Started
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up DVC: `dvc init`
4. Configure AWS credentials
5. Run training: `python train.py`

## Contributing
Please refer to the contribution guidelines in the repository for details on how to contribute to this project.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 