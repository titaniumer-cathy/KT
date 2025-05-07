Problem Formulation(Objective):
Knowledge tracing is a method of tracking personalised knowledge mastery and predict performance.

Given student's past exercise interactions X=(x1,x2,...,xn), predict some aspect of his/her next interaction xt+1. The interaction can be decomposed into xt=(qt,ct,rt), where qt is the question that the student attempts at timestamp t, ct is the concept that the question belongs to, and rt is the correctness of the student s answer. KT aims to predict whether the student will be able to answer the next exercise correctly, i.e., predict p(rt+1=1|et+1,xt).

In this project, it is specificly pointed out that we want to predict responses given concepts, so we are predicting p(rt+1=1|ct+1,xt)  or p(rt+1=1|ct+1,et+1,xt) instead of p(rt+1=1|et+1,xt) in traditional KT questions.

Methodology details

DKT (link: https://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf)
The original DKT algorithm firstly map question and response into interaction space of size 2*len(Q). And it output a sigmoid of size len(Q) representing the possibility of student answering each question correctly.

In order to predict responses given concepts, we replace the questions in DKT algorithm with concepts. It is a widely used KT assumption that all questions with the same concepts can be treated as a single question.(Context-Aware Attentive Knowledge Tracing) So the replacement has its actual meaning by using the same concepts assumption.

We keep other parts of the method the same as in the paper. So there is not much space to use questions_embedding and concepts_embedding in this method.

The results as followed:

Val Epoch: 3,   AUC: 0.8324132987206366,   Loss Mean: 0.4091655910015106,    ACC: 0.8393345315813445
Test AUC: 0.8192, Test Loss: 0.4355, Test Acc: 0.8240

SAKT ()
This model is a transformer-based self-attentive model for knowledge tracing.
We change the model a little bit to let the model output response probability based on concepts.
For the (query, key, value) in transformers, we let the query to be concepts_embedding, which is initialized with provided pretrained embeddings. Key and Value are concept-response pairs.

The results as followed:
QKV all in concepts: 
Val Epoch: 20,   AUC: 0.7951804341991286,   Loss Mean: 0.40735262632369995,    ACC: 0.82246763995749
Test AUC: 0.7854, Test Loss: 0.4321, Test Acc: 0.8080
QKV all in questions:
Val Epoch: 3,   AUC: 0.8484403158891555,   Loss Mean: 0.35861316323280334,    ACC: 0.8436550808817408
Test AUC: 0.8374, Test Loss: 0.3845, Test Acc: 0.8292

ATDKT(link: https://arxiv.org/pdf/2302.07942)

This paper propose using two Auxiliary Tasks to improve model performance : question tagging (QT) prediction task and individualized prior knowledge (IK) prediction task.
The QT task focuses on predicting the assigned KCs to the questions by modeling the intrinsic relations among the questions and KCs with students’ previous learning outcomes.
The IK task captures students’ global historical performance by progressively predicting studentlevel prior knowledge that is hidden in students’ historical learning interactions.

I think QT task is most helpful for this project. So I use QT loss along with traditional KT loss in this project.

Pretrained questions_embedding and concepts_embedding are used in this project. They are used to initialize the question embedding and concept embedding in atdkt model.
LSTM QT task
Val Epoch: 4,   AUC: 0.8451147264424443,   Loss Mean: 0.3550294516846902,    ACC: 0.8447540734718665
Test AUC: 0.8320, Test Loss: 0.3834, Test Acc: 0.8291

transformers QT task
Val Epoch: 5,   AUC: 0.8440720169122361,   Loss Mean: 0.3563565856343898,    ACC: 0.8441634384167455
Test AUC: 0.8305, Test Loss: 0.3851, Test Acc: 0.8287

Result Details
Training Details