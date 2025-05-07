# Problem Formulation(Objective):
Knowledge tracing is a method of tracking personalised knowledge mastery and predict performance.

Given student's past exercise interactions $X=(x_1,x_2,...,x_n)$, predict some aspect of his/her next interaction xt+1. The interaction can be decomposed into $x_t=(q_t,c_t,r_t)$, where $q_t$ is the question that the student attempts at timestamp t, $c_t$ is the concept that the question belongs to, and $r_t$ is the correctness of the student s answer. KT aims to predict whether the student will be able to answer the next exercise correctly, i.e., predict $P(r_{t+1}=1|q_{t+1},x_t)$.

In this project, it is specificly pointed out that we want to predict responses given concepts, so we are predicting $P(r_{t+1}=1|c_{t+1},x_t)$  or $P(r_{t+1}=1|c_{t+1},e_{t+1},xt)$ instead of $P(r_{t+1}=1|q_{t+1},x_t)$ in traditional KT questions.

# Methodology details

## [DKT](https://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf)
The original DKT algorithm firstly map question and response into interaction space of size 2*len(Q). And it output a sigmoid of size len(Q) representing the possibility of student answering each question correctly.

In order to predict responses given concepts, we replace the questions in DKT algorithm with concepts. It is a widely used KT assumption that all questions with the same concepts can be treated as a single question.(Context-Aware Attentive Knowledge Tracing) So the replacement has its actual meaning by using the same concepts assumption.

We keep other parts of the method the same as in the paper. So there is not much space to use questions_embedding and concepts_embedding in this method.

## [SAKT](https://arxiv.org/abs/1907.06837)
This model is a transformer-based self-attentive model for knowledge tracing.
We tried two versions of SAKT, one is the same as the paper, query is question embedding, key and value are question-response pairs, we call it sakt_question.

The other one is changed a little bit to let the model output probability based on concepts.For the (query, key, value) in transformers, we let the query to be concepts_embedding, which is initialized with provided pretrained embeddings. Key and Value are concept-response pairs. We call this version sakt_concept.

## [ATDKT](https://arxiv.org/pdf/2302.07942)

This paper propose using two Auxiliary Tasks to improve model performance : question tagging (QT) prediction task and individualized prior knowledge (IK) prediction task.
The QT task focuses on predicting the assigned KCs to the questions by modeling the intrinsic relations among the questions and KCs with students’ previous learning outcomes.
The IK task captures students’ global historical performance by progressively predicting studentlevel prior knowledge that is hidden in students’ historical learning interactions.

I think QT task is most helpful for this project. So I use QT loss along with traditional KT loss in this project.

Pretrained questions_embedding and concepts_embedding are used in this project. They are used to initialize the question embedding and concept embedding in atdkt model.

We tried two version of ATDKT, one is using LSTM for QT task, the other is using transformers in QT task. We name them atdkt_lstm and atdkt_transformer.


# Result Details

## Training Details

5 models are trained in this section:
1. dkt: using DKT architecture and use concept instead of question.
2. sakt_question: using SAKT architecture and is the same as the paper.
3. sakt_concept: use concept to replace question compared to sakt_question
4. atdkt_lstm: use QT task as auxiliary task and lstm in QT task.
5. atdkt_transformer: use QT task as auxiliary task and transformer in QT task.

The model is trained on fold 0-3 and validation data is fold 4 as default setting.

The config.json file contains most hyperparameter used in training.

1 Nvidia V100 card is used while training, each epoch of the training costs ~ 1 minute.

The validation accuracy and auc while train is below:
| ![validation accuracy](/images/VAL_Accuracy.jpeg "Validation Accuracy") | 
|:--:| 
| *Validation Accuracy* |

| !![validation auc](/images/Val_AUC.jpeg "Validation AUC") | 
|:--:| 
| *Validation AUC* |


## Best validation AUC achieved for each model is:

| Model             | Best validation AUC |
| ----------------- | ------------------- |
| dkt               | 0.8324              |
| sakt_question     | 0.8484              |
| sakt_concept      | 0.7952              |
| atdkt_lstm        | 0.8451              |
| atdkt_transformer | 0.8440              |


## Performance on test dataset
We used the model with highest validation auc during training process.

test performance for each model is:

| Model             | Test AUC | Test ACC |
| ----------------- | -------- | -------- |
| dkt               | 0.8192   | 0.8240   |
| sakt_question     | 0.8374   | 0.8292   |
| sakt_concept      | 0.7854   | 0.8080   |
| atdkt_lstm        | 0.8320   | 0.8291   |
| atdkt_transformer | 0.8305   | 0.8287   |


# Concluding

## Model analysis

1. Prediction Type
Of the five models only sakt_question is predicting $P(r_{t+1}=1|q_{t+1},x_t)$. dkt and sakt_concept is predicting $P(r_{t+1}=1|c_{t+1},x_t)$. atdkt_lstm and atdkt_transformer is predicting $P(r_{t+1}=1|q_{t+1},c_{t+1},x_t)$.
We can see sakt_question is having higher test performance compared to sakt_concept. From the training curve we can see sakt_concept is still growing at the end of epoch 20. It's really hard to learn the relationship between concept and response without any question information. SAKT is not a good method to adapt from response to concept directly.

2. ATDKT
atdkt is better than dkt and sakt_concept. I think this is attributed to the auciliary QT task which can capture the question and concepts relation. 

3. LSTM based model vs Transformer based model

LSTM is more lightweight. It is more suitable for small dataset with short sequence length. Otherwise there might be gradient vanishing or gradient explosion problems. LSTM is also computationally more affordable and can response fast for a real time application.

Transformer based model is good at capturing data sparsity. Long sequence length and large dataset size can make the transformer more stronger without worrying about gradient vanishing or gradient explosion problems. If we want to model the question that contains multimodal data(images, videos), transformers can also be able to handle that. But the computational cost of transformer is heavy both in training and inferencing.

4. Cold start
For a new question, we can first classify the problem as a certain knowledge concept and use the concept as a input to concept only models(dkt, sakt_concept for this implementation). So dkt and sakt_concept is more suitable for cold start models.

5. Interpretability
Considering interpretability, Transformers is a better choice because we can visualize the activated weights to let educators know the students' past performance that should paid attention to. 

## Potential Improvement
1. Hyperparameter tuning, expecially for transformer based models
2. Use knowledge embeddings, concepts embeddings as auciliary embeddings, instead of using as initialize embedding.
3. More modeling in question and concept relationship, e,g,[Context-Aware Attentive Knowledge Tracing](https://arxiv.org/pdf/2007.12324)
4. Data augmentation: For a long sequence, currently it is broken down to 1-200, 201-400. We can add 101-300 sequences as training data.
5. Add history prediction auciliary task from atdkt paper. We can also add history auciliary task and QT task in other model architectures.
6. Add response time as proposed in [SAINT+](https://arxiv.org/abs/2010.12042) as a input 
7. In atdkt, only add aggregation method is used, we can further explore mean aggregation and concat aggregation between embeddings.