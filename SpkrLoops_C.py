from functionsV2 import *
import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logging.getLogger().addHandler(logging.NullHandler())
logging.getLogger("natten.functional").setLevel(logging.ERROR)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

print("Device:", device)

DATASET_PATH = "../data"
CHECKPOINT_PATH = "../VIT/EMPR"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

wt_dc = 0.05

model_type = 'Dinat_Superloss'

fourclass = True
Speaker_Disentanglement = True
Entropy = True
pretrain = True
pretraining = False
new_test = False 

if pretrain==True:
    pathstr = r'/media/carol/Data/Documents/Emo_rec/NewMel/Pretraining/Domination'

    processor_path = os.path.join(pathstr, 'processor')
    model_path = os.path.join(pathstr, 'model')
    # model_path = os.path.join('G:\My Drive\MaSc\emo_rec\MODELS\For Paper\IEMOCAP\LABEL_CURRIUCULUM\PreSD_012', "model")
elif model_type != 'ViT':
    model_path = 'shi-labs/dinat-mini-in1k-224'
    processor_path = 'shi-labs/dinat-mini-in1k-224'
    pathstr = model_path
else:
    model_path = 'google/vit-base-patch16-224-in21k'
    processor_path = 'google/vit-base-patch16-224-in21k'
    pathstr = model_path

# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Check if CUDA is available and its version
if torch.cuda.is_available():
    print(f"CUDA is available. Version: {torch.version.cuda}")
    print(f"CUDA devices available: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available.")

print("HI")
EMOTIONS = {
    0: 'neutral',
    1: 'happy',
    2: 'sad',
    3: 'angry',
}
Map2Num = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'angry': 3,
}
Cval = 4


train_size = 20
eval_size = 20
from transformers import TrainingArguments, Trainer,  SchedulerType

metric_name = "eval_uar"
args = TrainingArguments(
    f".././logs2",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    disable_tqdm=True,  # This disables the default progress bar
    learning_rate=1e-5,
    lr_scheduler_type=SchedulerType.COSINE_WITH_RESTARTS,
    warmup_ratio=0.1,
    per_device_train_batch_size=train_size,
    per_device_eval_batch_size=eval_size,
    num_train_epochs=50,
    weight_decay=wt_dc,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='.././logs_DO_NOT_DELETE/3090',
    remove_unused_columns=False,
    # logging_strategy="epoch",  # Log at the end of each epoch

)

    
dataset_train = 'cairocode/IEMO_Mel_6' #'cairocode/IEMO_007_NOSPLIT'
train_d0 = load_dataset(dataset_train, split='train')

dataset_val = 'cairocode/MSPI_Mel6'#'cairocode/MSPI_007_NOSPLIT_an'
val_dataset0 = load_dataset(dataset_val, split='train')
val_dataset0  = val_dataset0.filter(filter_m_examples)
ds_tr = os.path.split(dataset_train)[1]
ds_vl = os.path.split(dataset_val)[1]

base_dir = os.path.join(r'/media/carol/Data/Documents/Emo_rec/LOSO', ds_tr)
if pretrain == True and Speaker_Disentanglement == True:
    base_dir = os.path.join(base_dir, "PRSD")
elif pretrain == True :
    base_dir = os.path.join(base_dir, "PR")
elif Speaker_Disentanglement == True :
    base_dir = os.path.join(base_dir, "SD")
else:
    base_dir = os.path.join(base_dir, "OG")
root_path = create_unique_output_dir(base_dir)
print(root_path)
 
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from collections import defaultdict
# Example usage

log_file = os.path.join(root_path,"training_logs.csv")
fieldnames = ["timestamp", "log"]


spkrs = [sample['speakerID'] for sample in train_d0]
unique_speakers = list(set(spkrs))
print(unique_speakers)

all_y_true = defaultdict(list)
all_y_pred = defaultdict(list)

total_results = pd.DataFrame()

xcorp_results = {
    'y_true': [],
    'y_pred': {f'run_{i}': [] for i in range(10)}
}
for i in range (0,len(unique_speakers), 1):
    
    speakers = [unique_speakers[i]]  #    speakers = [937+i]
    # speakers = [11-unique_speakers[i]]

    num = speakers[0] -1 #-1
    print(f"NUM = {num} __ SPKRS {speakers}")
    print(f"\n {'#'*120}")
    print(f"                                          STARTING SPEAKER {num} & {num+1}                                                      ")
    print(f"\n {'#'*120}")

    new_model_path = os.path.join(root_path, str(num))
    os.makedirs(new_model_path, exist_ok=True)
    
    # Create the test split
    test_dataset = train_d0.filter(lambda x: x['speakerID'] in speakers).filter(filter_m_examples)
    
    # Create the remaining data
    train_set = train_d0.filter(lambda x: x['speakerID'] not in speakers).filter(filter_m_examples)
    train_set = balance_dataset(train_set, label_column="label", seed=32)

    # train_set = train_set.train_test_split(test_size = 0.2)
    train_dataset = train_set #['train']
    val_dataset = test_dataset #train_set['test']
    # val_dataset = concatenate_datasets([val_dataset0, test_dataset])

    processor = AutoImageProcessor.from_pretrained(processor_path)

    base_model = DinatForImageClassification.from_pretrained(
        model_path,
        id2label=EMOTIONS,
        num_labels=Cval,
        label2id=Map2Num,
        ignore_mismatched_sizes=True,
        problem_type='single_label_classification',
        output_hidden_states=True
    )
    if Speaker_Disentanglement == True:
        custom_dataset = CustomDataset(train_dataset)
        custom_sampler = CustomSampler(custom_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=custom_sampler, collate_fn=collate_fn, batch_size=train_size)
    else:
        train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=train_size)

    class_weights = [1.1, 2,1, 1]
    class_weights = torch.tensor(class_weights)
 
    alpha = 1
    beta = 0.08
    center = 1.6
    model = CustomDinatForImageClassification_V2(
        base_model=base_model,
        num_classes=Cval,          # Number of emotion classes
        feature_dim=512,        # Feature dimension for ViT (check your model's specs)
        class_weights=class_weights,  # Optional: weights for each class
        alpha=alpha,
        beta=beta,
        center_lr=center
    )

    model.to(device)
    super_loss_params = {
    'C': Cval,  # Example value, adjust based on your needs
    'lam': 0.01,  # Example value, adjust based on your needs
    'batch_size': args.train_batch_size,  # Pass the batch size dynamically
    'class_weights':class_weights
    }

    
    val_dataset.set_transform(val_transforms)
    val_dataset0.set_transform(val_transforms)
    train_dataset.set_transform(train_transforms)

    

    # early_stopping = EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.001)
    early_stopping = EarlyStoppingCallback(early_stopping_patience=12, early_stopping_threshold=0.001)

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
        callbacks=[early_stopping]#, ClassWeightLoggerCallback()],  # Add the new callback here
        # super_loss_params=super_loss_params,  # Pass the custom loss parameters here
        )
    trainer.train()
    trainer.save_model(os.path.join(new_model_path, "model"))
    processor.save_pretrained(os.path.join(new_model_path, "processor"))
    test_dataset.set_transform(test_transforms)
    outputs = trainer.predict(test_dataset)
    print(outputs.metrics['test_accuracy']*100, "\t", outputs.metrics['test_uar']*100)
    print(outputs.metrics)
    model_info = {
        "Pretrain_file": pathstr,
        "Dataset Used": dataset_train,
        "Model Type": model_type,
        "Super Loss PARAMS": super_loss_params,
        "Speaker Disentanglement": Speaker_Disentanglement,
        "Entropy Curriculum Training": Entropy,
        "Column Trained on": column,
        "Test Results": outputs.metrics,
        "Test SpeakerID": speakers,
        "Class Weight": class_weights,
        "NUM":  num,
        "Weight decay": wt_dc,
        "alpha": alpha,
        "beta": beta,
        "center": center,
    }
    file_path = save_model_header(new_model_path, model_info)
    matrix_path = save_confusion_matrix(outputs, dataset_train, new_model_path, Map2Num)
    dataset_test = dataset_val
    outputs2 = trainer.predict(val_dataset0)
    # print(outputs2.metrics)
    # print(outputs2.metrics['test_accuracy']*100, "\t", outputs2.metrics['test_uar']*100)

    new_row = {f'{ds_tr}_ACC': outputs.metrics['test_accuracy']*100, f'{ds_tr}_UAR': outputs.metrics['test_uar']*100, f'{ds_vl}_ACC': outputs2.metrics['test_accuracy']*100,f'{ds_vl}_UAR': outputs2.metrics['test_uar']*100}
    print(f"\n {'-'*120}")
    print("\n\n", new_row, "\n")
    
    # Method 1: Using concat()
    total_results = pd.concat([total_results, pd.DataFrame([new_row])], ignore_index=True)

    if i == 0:
        xcorp_results['y_true'].extend(outputs2.label_ids)

    # Store y_pred for each run
    xcorp_results['y_pred'][f'run_{i}'].extend(outputs2.predictions.argmax(axis=1))


    matrix_path = save_confusion_matrix(outputs2, dataset_test, new_model_path, Map2Num)
    all_y_true[ds_tr].extend(outputs.label_ids)
    all_y_pred[ds_tr].extend(outputs.predictions.argmax(axis=1))
    all_y_true[ds_vl].extend(outputs2.label_ids)
    all_y_pred[ds_vl].extend(outputs2.predictions.argmax(axis=1))



    torch.cuda.empty_cache()
    del trainer
    del model

# If you want to get the final prediction based on majority voting across all runs
y_true = np.array(xcorp_results['y_true'])

final_prediction = np.array([np.bincount([xcorp_results['y_pred'][f'run_{i}'][j] for i in range(0,10,1)]).argmax() 
                             for j in range(len(y_true))])

final_accuracy = accuracy_score(y_true, final_prediction)
print(f"Final accuracy after majority voting: {final_accuracy}")
print(total_results)
avg_results = total_results.mean(numeric_only=True)

print("\nAVERAGE RESULTS\n", avg_results)

final_metrics = {}
for dataset in [ds_tr, ds_vl]:
    y_true = np.array(all_y_true[dataset])
    y_pred = np.array(all_y_pred[dataset])
    
    acc = accuracy_score(y_true, y_pred) * 100
    uar = balanced_accuracy_score(y_true, y_pred) * 100
    
    final_metrics[f'{dataset}_ACC'] = acc
    final_metrics[f'{dataset}_UAR'] = uar

print("\nFinal Metrics:")
print(final_metrics)
model_info = {
    "Pretrain_file": pathstr,
    "Dataset Used": dataset_train,
    "Model Type": model_type,
    "Super Loss PARAMS": super_loss_params,
    "Speaker Disentanglement": Speaker_Disentanglement,
    "Entropy Curriculum Training": Entropy,
    "Column Trained on": column,
    "Test SpeakerID": speakers,
    "Weight decay": wt_dc, 
    "Avg Results": avg_results, 
    "total results": final_metrics,
    "alpha": alpha,
    "beta": beta,
    "center": center,
}
file_path = save_model_header(root_path, model_info)



new_model_path = os.path.join(root_path, str("FULL_DS_Xcorp"))
os.makedirs(new_model_path, exist_ok=True)

# Create the test split
test_dataset = val_dataset0

train_dataset = train_d0 #['train']
val_dataset = test_dataset #train_set['test']
# val_dataset = concatenate_datasets([val_dataset0, test_dataset])

processor = AutoImageProcessor.from_pretrained(processor_path)

base_model = DinatForImageClassification.from_pretrained(
    model_path,
    id2label=EMOTIONS,
    num_labels=Cval,
    label2id=Map2Num,
    ignore_mismatched_sizes=True,
    problem_type='single_label_classification',
    output_hidden_states=True
)
if Speaker_Disentanglement == True:
    custom_dataset = CustomDataset(train_dataset)
    custom_sampler = CustomSampler(custom_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=custom_sampler, collate_fn=collate_fn, batch_size=train_size)
else:
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=train_size)

class_weights = [1.1, 2,1, 1]

alpha = 1
beta = 0.08
center = 1.6
model = CustomDinatForImageClassification_V2(
    base_model=base_model,
    num_classes=Cval,          # Number of emotion classes
    feature_dim=512,        # Feature dimension for ViT (check your model's specs)
    class_weights=class_weights,  # Optional: weights for each class
    alpha=alpha,
    beta=beta,
    center_lr=center
)

model.to(device)
super_loss_params = {
'C': Cval,  # Example value, adjust based on your needs
'lam': 0.01,  # Example value, adjust based on your needs
'batch_size': args.train_batch_size,  # Pass the batch size dynamically
'class_weights':class_weights
}


val_dataset.set_transform(val_transforms)
val_dataset0.set_transform(val_transforms)
train_dataset.set_transform(train_transforms)

# early_stopping = EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.001)
early_stopping = EarlyStoppingCallback(early_stopping_patience=12, early_stopping_threshold=0.001)

trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
    callbacks=[early_stopping]#, ClassWeightLoggerCallback()],  # Add the new callback here
    # super_loss_params=super_loss_params,  # Pass the custom loss parameters here
    )
trainer.train()
trainer.save_model(os.path.join(new_model_path, "model"))
processor.save_pretrained(os.path.join(new_model_path, "processor"))
test_dataset.set_transform(test_transforms)
outputs = trainer.predict(test_dataset)
print(outputs.metrics['test_accuracy']*100, "\t", outputs.metrics['test_uar']*100)
print(outputs.metrics)
model_info = {
    "Pretrain_file": pathstr,
    "Dataset Used": dataset_train,
    "Model Type": model_type,
    "Super Loss PARAMS": super_loss_params,
    "Speaker Disentanglement": Speaker_Disentanglement,
    "Entropy Curriculum Training": Entropy,
    "Column Trained on": column,
    "Test Results": outputs.metrics,
    "Test SpeakerID": speakers,
    "Class Weight": class_weights,
    "NUM":  num,
    "Weight decay": wt_dc,
    "alpha": alpha,
    "beta": beta,
    "center": center,
}
file_path = save_model_header(new_model_path, model_info)
matrix_path = save_confusion_matrix(outputs, dataset_train, new_model_path, Map2Num)