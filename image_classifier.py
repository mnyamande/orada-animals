import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50, InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import pandas as pd
import matplotlib.pyplot as plt
import time

# Set page config
st.set_page_config(
    page_title="Animal Image Classifier",
    page_icon="🐾",
    layout="wide"
)

# App title and description
st.title("Animal Image Classification 🐾")
st.markdown("""
This application uses pre-trained deep learning models to classify uploaded animal images.
Upload an image of an animal, and the model will predict what animal it is!
""")

# Sidebar for model selection
st.sidebar.title("Model Settings")
model_name = st.sidebar.selectbox(
    "Choose a model",
    ["MobileNetV2", "ResNet50", "InceptionV3"]
)

# Confidence threshold slider
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.05,
    help="Show only predictions above this confidence level"
)

# Add option to show animal categories
show_categories = st.sidebar.checkbox("Show all animal categories", value=False)

# Function to load model
@st.cache_resource
def load_model(model_name):
    if model_name == "MobileNetV2":
        model = MobileNetV2(weights='imagenet')
        preprocess = mobilenet_preprocess
        input_size = (224, 224)
    elif model_name == "ResNet50":
        model = ResNet50(weights='imagenet')
        preprocess = resnet_preprocess
        input_size = (224, 224)
    elif model_name == "InceptionV3":
        model = InceptionV3(weights='imagenet')
        preprocess = inception_preprocess
        input_size = (299, 299)
    return model, preprocess, input_size

# Function to preprocess image
def preprocess_image(uploaded_file, input_size, preprocess_func):
    img = Image.open(uploaded_file).convert('RGB')
    st.sidebar.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Resize image to the required input size
    img = img.resize(input_size)
    
    # Convert to array and preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_func(img_array)
    
    return processed_img, img

# Function to get animal classes
def get_animal_classes():
    # ImageNet classes that are animals (rough selection)
    animal_categories = {
        'n01440764': 'tench',
        'n01443537': 'goldfish',
        'n01484850': 'great white shark',
        'n01491361': 'tiger shark',
        'n01494475': 'hammerhead shark',
        'n01496331': 'electric ray',
        'n01498041': 'stingray',
        'n01514668': 'cock',
        'n01514859': 'hen',
        'n01518878': 'ostrich',
        'n01530575': 'brambling',
        'n01531178': 'goldfinch',
        'n01532829': 'house finch',
        'n01534433': 'junco',
        'n01537544': 'indigo bunting',
        'n01558993': 'robin',
        'n01560419': 'bulbul',
        'n01580077': 'jay',
        'n01582220': 'magpie',
        'n01592084': 'chickadee',
        'n01601694': 'water ouzel',
        'n01608432': 'kite',
        'n01614925': 'bald eagle',
        'n01616318': 'vulture',
        'n01622779': 'great grey owl',
        'n01629819': 'European fire salamander',
        'n01630670': 'common newt',
        'n01631663': 'eft',
        'n01632458': 'spotted salamander',
        'n01632777': 'axolotl',
        'n01641577': 'bullfrog',
        'n01644373': 'tree frog',
        'n01644900': 'tailed frog',
        'n01664065': 'loggerhead',
        'n01665541': 'leatherback turtle',
        'n01667114': 'mud turtle',
        'n01667778': 'terrapin',
        'n01669191': 'box turtle',
        'n01675722': 'banded gecko',
        'n01677366': 'common iguana',
        'n01682714': 'American chameleon',
        'n01685808': 'whiptail',
        'n01687978': 'agama',
        'n01688243': 'frilled lizard',
        'n01689811': 'alligator lizard',
        'n01692333': 'Gila monster',
        'n01693334': 'green lizard',
        'n01694178': 'African chameleon',
        'n01695060': 'Komodo dragon',
        'n01697457': 'African crocodile',
        'n01698640': 'American alligator',
        'n01704323': 'triceratops',
        'n01728572': 'thunder snake',
        'n01728920': 'ringneck snake',
        'n01729322': 'hognose snake',
        'n01729977': 'green snake',
        'n01734418': 'king snake',
        'n01735189': 'garter snake',
        'n01737021': 'water snake',
        'n01739381': 'vine snake',
        'n01740131': 'night snake',
        'n01742172': 'boa constrictor',
        'n01744401': 'rock python',
        'n01748264': 'Indian cobra',
        'n01749939': 'green mamba',
        'n01751748': 'sea snake',
        'n01753488': 'horned viper',
        'n01755581': 'diamondback',
        'n01756291': 'sidewinder',
        'n01768244': 'trilobite',
        'n01770081': 'harvestman',
        'n01770393': 'scorpion',
        'n01773157': 'black and gold garden spider',
        'n01773549': 'barn spider',
        'n01773797': 'garden spider',
        'n01774384': 'black widow',
        'n01774750': 'tarantula',
        'n01775062': 'wolf spider',
        'n01776313': 'tick',
        'n01784675': 'centipede',
        'n01795545': 'black grouse',
        'n01796340': 'ptarmigan',
        'n01797886': 'ruffed grouse',
        'n01798484': 'prairie chicken',
        'n01806143': 'peacock',
        'n01806567': 'quail',
        'n01807496': 'partridge',
        'n01817953': 'African grey',
        'n01818515': 'macaw',
        'n01819313': 'sulphur-crested cockatoo',
        'n01820546': 'lorikeet',
        'n01824575': 'coucal',
        'n01828970': 'bee eater',
        'n01829413': 'hornbill',
        'n01833805': 'hummingbird',
        'n01843065': 'jacamar',
        'n01843383': 'toucan',
        'n01847000': 'drake',
        'n01855032': 'red-breasted merganser',
        'n01855672': 'goose',
        'n01860187': 'black swan',
        'n01871265': 'tusker',
        'n01872401': 'echidna',
        'n01873310': 'platypus',
        'n01877812': 'wallaby',
        'n01882714': 'koala',
        'n01883070': 'wombat',
        'n01910747': 'jellyfish',
        'n01914609': 'sea anemone',
        'n01917289': 'brain coral',
        'n01924916': 'flatworm',
        'n01930112': 'nematode',
        'n01943899': 'conch',
        'n01944390': 'snail',
        'n01945685': 'slug',
        'n01950731': 'sea slug',
        'n01955084': 'chiton',
        'n01968897': 'chambered nautilus',
        'n01978287': 'Dungeness crab',
        'n01978455': 'rock crab',
        'n01980166': 'fiddler crab',
        'n01981276': 'king crab',
        'n01983481': 'American lobster',
        'n01984695': 'spiny lobster',
        'n01985128': 'crayfish',
        'n01986214': 'hermit crab',
        'n01990800': 'isopod',
        'n02002556': 'white stork',
        'n02002724': 'black stork',
        'n02006656': 'spoonbill',
        'n02007558': 'flamingo',
        'n02009229': 'little blue heron',
        'n02009912': 'American egret',
        'n02011460': 'bittern',
        'n02012849': 'crane',
        'n02013706': 'limpkin',
        'n02017213': 'European gallinule',
        'n02018207': 'American coot',
        'n02018795': 'bustard',
        'n02025239': 'ruddy turnstone',
        'n02027492': 'red-backed sandpiper',
        'n02028035': 'redshank',
        'n02033041': 'dowitcher',
        'n02037110': 'oystercatcher',
        'n02051845': 'pelican',
        'n02056570': 'king penguin',
        'n02058221': 'albatross',
        'n02066245': 'grey whale',
        'n02071294': 'killer whale',
        'n02074367': 'dugong',
        'n02077923': 'sea lion',
        'n02085620': 'Chihuahua',
        'n02085782': 'Japanese spaniel',
        'n02085936': 'Maltese dog',
        'n02086079': 'Pekinese',
        'n02086240': 'Shih-Tzu',
        'n02086646': 'Blenheim spaniel',
        'n02086910': 'papillon',
        'n02087046': 'toy terrier',
        'n02087394': 'Rhodesian ridgeback',
        'n02088094': 'Afghan hound',
        'n02088238': 'basset',
        'n02088364': 'beagle',
        'n02088466': 'bloodhound',
        'n02088632': 'bluetick',
        'n02089078': 'black-and-tan coonhound',
        'n02089867': 'Walker hound',
        'n02089973': 'English foxhound',
        'n02090379': 'redbone',
        'n02090622': 'borzoi',
        'n02090721': 'Irish wolfhound',
        'n02091032': 'Italian greyhound',
        'n02091134': 'whippet',
        'n02091244': 'Ibizan hound',
        'n02091467': 'Norwegian elkhound',
        'n02091635': 'otterhound',
        'n02091831': 'Saluki',
        'n02092002': 'Scottish deerhound',
        'n02092339': 'Weimaraner',
        'n02093256': 'Staffordshire bullterrier',
        'n02093428': 'American Staffordshire terrier',
        'n02093647': 'Bedlington terrier',
        'n02093754': 'Border terrier',
        'n02093859': 'Kerry blue terrier',
        'n02093991': 'Irish terrier',
        'n02094114': 'Norfolk terrier',
        'n02094258': 'Norwich terrier',
        'n02094433': 'Yorkshire terrier',
        'n02095314': 'wire-haired fox terrier',
        'n02095570': 'Lakeland terrier',
        'n02095889': 'Sealyham terrier',
        'n02096051': 'Airedale',
        'n02096177': 'cairn',
        'n02096294': 'Australian terrier',
        'n02096437': 'Dandie Dinmont',
        'n02096585': 'Boston bull',
        'n02097047': 'miniature schnauzer',
        'n02097130': 'giant schnauzer',
        'n02097209': 'standard schnauzer',
        'n02097298': 'Scotch terrier',
        'n02097474': 'Tibetan terrier',
        'n02097658': 'silky terrier',
        'n02098105': 'soft-coated wheaten terrier',
        'n02098286': 'West Highland white terrier',
        'n02098413': 'Lhasa',
        'n02099267': 'flat-coated retriever',
        'n02099429': 'curly-coated retriever',
        'n02099601': 'golden retriever',
        'n02099712': 'Labrador retriever',
        'n02099849': 'Chesapeake Bay retriever',
        'n02100236': 'German short-haired pointer',
        'n02100583': 'vizsla',
        'n02100735': 'English setter',
        'n02100877': 'Irish setter',
        'n02101006': 'Gordon setter',
        'n02101388': 'Brittany spaniel',
        'n02101556': 'clumber',
        'n02102040': 'English springer',
        'n02102177': 'Welsh springer spaniel',
        'n02102318': 'cocker spaniel',
        'n02102480': 'Sussex spaniel',
        'n02102973': 'Irish water spaniel',
        'n02104029': 'kuvasz',
        'n02104365': 'schipperke',
        'n02105056': 'groenendael',
        'n02105162': 'malinois',
        'n02105251': 'briard',
        'n02105412': 'kelpie',
        'n02105505': 'komondor',
        'n02105641': 'Old English sheepdog',
        'n02105855': 'Shetland sheepdog',
        'n02106030': 'collie',
        'n02106166': 'Border collie',
        'n02106382': 'Bouvier des Flandres',
        'n02106550': 'Rottweiler',
        'n02106662': 'German shepherd',
        'n02107142': 'Doberman',
        'n02107312': 'miniature pinscher',
        'n02107574': 'Greater Swiss Mountain dog',
        'n02107683': 'Bernese mountain dog',
        'n02107908': 'Appenzeller',
        'n02108000': 'EntleBucher',
        'n02108089': 'boxer',
        'n02108422': 'bull mastiff',
        'n02108551': 'Tibetan mastiff',
        'n02108915': 'French bulldog',
        'n02109047': 'Great Dane',
        'n02109525': 'Saint Bernard',
        'n02109961': 'Eskimo dog',
        'n02110063': 'malamute',
        'n02110185': 'Siberian husky',
        'n02110341': 'dalmatian',
        'n02110627': 'affenpinscher',
        'n02110806': 'basenji',
        'n02110958': 'pug',
        'n02111129': 'Leonberg',
        'n02111277': 'Newfoundland',
        'n02111500': 'Great Pyrenees',
        'n02111889': 'Samoyed',
        'n02112018': 'Pomeranian',
        'n02112137': 'chow',
        'n02112350': 'keeshond',
        'n02112706': 'Brabancon griffon',
        'n02113023': 'Pembroke',
        'n02113186': 'Cardigan',
        'n02113624': 'toy poodle',
        'n02113712': 'miniature poodle',
        'n02113799': 'standard poodle',
        'n02113978': 'Mexican hairless',
        'n02114367': 'timber wolf',
        'n02114548': 'white wolf',
        'n02114712': 'red wolf',
        'n02114855': 'coyote',
        'n02115641': 'dingo',
        'n02115913': 'dhole',
        'n02116738': 'African hunting dog',
        'n02117135': 'hyena',
        'n02119022': 'red fox',
        'n02119789': 'kit fox',
        'n02120079': 'Arctic fox',
        'n02120505': 'grey fox',
        'n02123045': 'tabby',
        'n02123159': 'tiger cat',
        'n02123394': 'Persian cat',
        'n02123597': 'Siamese cat',
        'n02124075': 'Egyptian cat',
        'n02125311': 'cougar',
        'n02127052': 'lynx',
        'n02128385': 'leopard',
        'n02128757': 'snow leopard',
        'n02128925': 'jaguar',
        'n02129165': 'lion',
        'n02129604': 'tiger',
        'n02130308': 'cheetah',
        'n02132136': 'brown bear',
        'n02133161': 'American black bear',
        'n02134084': 'ice bear',
        'n02134418': 'sloth bear',
        'n02137549': 'mongoose',
        'n02138441': 'meerkat',
        'n02165105': 'tiger beetle',
        'n02165456': 'ladybug',
        'n02167151': 'ground beetle',
        'n02168699': 'long-horned beetle',
        'n02169497': 'leaf beetle',
        'n02172182': 'dung beetle',
        'n02174001': 'rhinoceros beetle',
        'n02177972': 'weevil',
        'n02190166': 'fly',
        'n02206856': 'bee',
        'n02219486': 'ant',
        'n02226429': 'grasshopper',
        'n02229544': 'cricket',
        'n02231487': 'walking stick',
        'n02233338': 'cockroach',
        'n02236044': 'mantis',
        'n02256656': 'cicada',
        'n02259212': 'leafhopper',
        'n02264363': 'lacewing',
        'n02268443': 'dragonfly',
        'n02268853': 'damselfly',
        'n02276258': 'admiral',
        'n02277742': 'ringlet',
        'n02279972': 'monarch',
        'n02280649': 'cabbage butterfly',
        'n02281406': 'sulphur butterfly',
        'n02281787': 'lycaenid',
        'n02317335': 'starfish',
        'n02319095': 'sea urchin',
        'n02321529': 'sea cucumber',
        'n02325366': 'wood rabbit',
        'n02326432': 'hare',
        'n02328150': 'Angora',
        'n02342885': 'hamster',
        'n02346627': 'porcupine',
        'n02356798': 'fox squirrel',
        'n02361337': 'marmot',
        'n02363005': 'beaver',
        'n02364673': 'guinea pig',
        'n02389026': 'sorrel',
        'n02391049': 'zebra',
        'n02395406': 'hog',
        'n02396427': 'wild boar',
        'n02397096': 'warthog',
        'n02398521': 'hippopotamus',
        'n02403003': 'ox',
        'n02408429': 'water buffalo',
        'n02410509': 'bison',
        'n02412080': 'ram',
        'n02415577': 'bighorn',
        'n02417914': 'ibex',
        'n02422106': 'hartebeest',
        'n02422699': 'impala',
        'n02423022': 'gazelle',
        'n02437312': 'Arabian camel',
        'n02437616': 'llama',
        'n02441942': 'weasel',
        'n02442845': 'mink',
        'n02443114': 'polecat',
        'n02443484': 'black-footed ferret',
        'n02444819': 'otter',
        'n02445715': 'skunk',
        'n02447366': 'badger',
        'n02454379': 'armadillo',
        'n02457408': 'three-toed sloth',
        'n02480495': 'orangutan',
        'n02480855': 'gorilla',
        'n02481823': 'chimpanzee',
        'n02483362': 'gibbon',
        'n02483708': 'siamang',
        'n02484975': 'guenon',
        'n02486261': 'patas',
        'n02486410': 'baboon',
        'n02487347': 'macaque',
        'n02488291': 'langur',
        'n02488702': 'colobus',
        'n02489166': 'proboscis monkey',
        'n02490219': 'marmoset',
        'n02492035': 'capuchin',
        'n02492660': 'howler monkey',
        'n02493509': 'titi',
        'n02493793': 'spider monkey',
        'n02494079': 'squirrel monkey',
        'n02497673': 'Madagascar cat',
        'n02500267': 'indri',
        'n02504013': 'Indian elephant',
        'n02504458': 'African elephant',
        'n02509815': 'lesser panda',
        'n02510455': 'giant panda',
        'n02514041': 'barracouta',
        'n02526121': 'eel',
        'n02536864': 'coho',
        'n02606052': 'rock beauty',
        'n02607072': 'anemone fish',
        'n02640242': 'sturgeon',
        'n02641379': 'gar',
        'n02643566': 'lionfish',
        'n02655020': 'puffer',
        'n02690373': 'airliner',
    }
    return animal_categories

# Function to predict the image
def predict_image(model, processed_img, animal_categories):
    # Make prediction
    preds = model.predict(processed_img)
    
    # Get ImageNet class predictions
    from tensorflow.keras.applications.imagenet_utils import decode_predictions
    predictions = decode_predictions(preds, top=10)[0]
    
    # Filter for only animal predictions
    animal_preds = []
    for pred in predictions:
        if pred[0] in animal_categories:
            animal_preds.append((pred[0], animal_categories[pred[0]], pred[2]))
    
    return animal_preds

# Main application
def main():
    # File uploader
    uploaded_file = st.file_uploader("Choose an animal image...", type=["jpg", "jpeg", "png"])
    
    # Show all animal categories when checkbox is selected
    if show_categories:
        animal_categories = get_animal_classes()
        with st.expander("All Animal Categories"):
            # Convert to DataFrame for better display
            categories_df = pd.DataFrame(list(animal_categories.items()), columns=['Class ID', 'Class Name'])
            st.dataframe(categories_df, height=300)
    
    if uploaded_file is not None:
        # Progress bar for model loading
        with st.spinner(f"Loading {model_name} model..."):
            model, preprocess, input_size = load_model(model_name)
            st.success(f"{model_name} loaded successfully!")
        
        # Progress bar for processing image
        with st.spinner("Processing image..."):
            processed_img, original_img = preprocess_image(uploaded_file, input_size, preprocess)
            animal_categories = get_animal_classes()
        
        # Prediction with progress bar
        with st.spinner("Making prediction..."):
            # Add slight delay to show progress bar (for educational purposes)
            time.sleep(1)  
            animal_preds = predict_image(model, processed_img, animal_categories)
        
        # Display results
        st.subheader("Prediction Results")
        
        # Create columns for image and predictions
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(original_img, caption="Analyzed Image", use_column_width=True)
        
        with col2:
            if animal_preds:
                # Display predictions as a table
                results_df = pd.DataFrame(
                    [(name, f"{confidence*100:.2f}%") for _, name, confidence in animal_preds if confidence >= confidence_threshold],
                    columns=["Animal", "Confidence"]
                )
                st.table(results_df)
                
                # Create bar chart for top predictions
                filtered_preds = [(name, confidence) for _, name, confidence in animal_preds if confidence >= confidence_threshold]
                
                if filtered_preds:
                    chart_data = pd.DataFrame(
                        filtered_preds,
                        columns=["Animal", "Confidence"]
                    )
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(chart_data["Animal"], chart_data["Confidence"])
                    ax.set_xlabel("Confidence")
                    ax.set_title("Top Animal Predictions")
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.warning("No animal detected in the image with sufficient confidence. Try lowering the confidence threshold or uploading a different image.")
        
        # Educational component - explain the model
        with st.expander("How does this model work?"):
            st.markdown(f"""
            ### About {model_name}
            
            This application uses {model_name}, a deep learning model pre-trained on the ImageNet dataset, 
            which contains over 1.2 million images across 1,000 categories.
            
            **The prediction process:**
            
            1. **Image Pre-processing**: The uploaded image is resized to {input_size[0]}x{input_size[1]} pixels and normalized.
            2. **Feature Extraction**: The model extracts visual features from the image using convolutional layers.
            3. **Classification**: These features are passed through fully connected layers to predict the animal class.
            4. **Filtering**: We filter the results to show only animal categories with confidence above your selected threshold.
            
            For your workshop, you can discuss concepts like transfer learning, convolutional neural networks, and image preprocessing.
            """)
    
    else:
        # Display sample images and information when no file is uploaded
        st.info("Please upload an image to get started!")
        st.markdown("""
        ### Tips for best results:
        - Use clear, well-lit images
        - Center the animal in the frame
        - Try different models for comparison
        - Adjust the confidence threshold based on results
        """)

if __name__ == "__main__":
    main()
