import streamlit as st
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import os
from io import BytesIO
import time

#page configuration
st.set_page_config(
    page_title="Bird Species Classifier",page_icon="🪶",layout="wide",initial_sidebar_state="expanded")
#styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;0,700;1,400;1,600&family=JetBrains+Mono:wght@300;400;500&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap');
:root {
    --parchment:  cornsilk;
    --aged:       burlywood;
    --ink:        saddlebrown;
    --deep-ink:   maroon;
    --sky:        steelblue;
    --dawn:       peru;
    --moss:       darkolivegreen;
    --feather:    tan;
    --shadow:     rgba(101,67,33,0.15);
}

html, body, .stApp {
    background: cornsilk !important;
    font-family: 'Libre Baskerville', Georgia, serif;
}

section[data-testid="stAppViewContainer"] {
    background: cornsilk !important;
}

section[data-testid="stAppViewContainer"]::before {
    content:'';
    position:fixed;
    inset: 0;
    background-image:
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 28px,
            rgba(101,67,33,0.04) 28px,
            rgba(101,67,33,0.04) 29px
        );
    pointer-events: none;
    z-index: 0;
}
.block-container {
    padding-top: 0 !important;
    padding-bottom: 3rem;
    position: relative;
    z-index: 1;
}
[data-testid="stSidebar"] {
    background: antiquewhite !important;
    border-right: 2px solid burlywood;
}
[data-testid="stSidebar"] * { color: saddlebrown !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Crimson Pro', Georgia, serif !important;
    color: maroon !important;
    letter-spacing: 0.04em;
}
[data-testid="stSidebar"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: saddlebrown !important;
}

h1, h2, h3 {
    font-family: 'Crimson Pro', Georgia, serif !important;
    color: maroon !important;
}
h1 { letter-spacing: -0.01em; }
p, li, span { color: saddlebrown; }
label { color: saddlebrown !important; }
.stMarkdown { color: saddlebrown !important; }

[data-testid="stFileUploader"] {
    background: antiquewhite;
    border: 2px dashed burlywood;
    border-radius: 4px;
    padding: 1.5rem;
}
[data-testid="stFileUploader"] * { color: saddlebrown !important; }
[data-testid="stFileUploader"]:hover {
    border-color: peru;
    background: linen;
}

.stButton > button {
    background: #ffcccb !important; 
    color: #4a0000 !important;      
    border: none;
    border-radius: 3px;
    font-family: 'Crimson Pro', Georgia, serif;
    font-size: 1.05rem;
    font-weight: 700;               
    letter-spacing: 0.06em;
    padding: 0.65rem 1.8rem;
    transition: all 0.25s ease;
    box-shadow: 3px 3px 0 #8b0000;
    width: 100%;
}
.stButton > button:hover {
    background: #ff8c8c !important;
    color: #2b0000 !important;     
    transform: translate(-1px, -1px);
    box-shadow: 4px 4px 0 #5e0000;
}
.stButton > button:active {
    transform: translate(1px, 1px);
    box-shadow: 1px 1px 0 #5e0000;
}

[data-testid="stExpander"] {
    background: antiquewhite;
    border: 1px solid burlywood;
    border-radius: 4px;
}
[data-testid="stExpander"] summary {
    font-family: 'Crimson Pro', Georgia, serif;
    color: maroon !important;
    font-size: 1rem;
    font-weight: 600;
}

.stSpinner > div { border-top-color: maroon !important; }

hr { border-color: burlywood; opacity: 0.5; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: cornsilk; }
::-webkit-scrollbar-thumb { background: burlywood; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


#bird data
BIRD_INFO = {
    'AMERICAN GOLDFINCH': {
        'scientific':'Spinus tristis',
        'habitat': 'Open fields, meadows & gardens across North America',
        'diet': 'Seeds especially thistle and sunflower',
        'size':'11–13 cm · 11–20 g',
        'wingspan':  '19–22 cm',
        'fact': 'Males undergo a dramatic moult each spring, turning vivid canary yellow to attract mates.',
        'silhouette':  '🐦','color_band':'gold',},
    'BARN OWL': {
        'scientific': 'Tyto alba',
        'habitat': 'Farmlands, open grasslands & woodland edges. Found on every continent except Antarctica',
        'diet': 'Small mammals, primarily voles, mice and shrews',
        'size': '33–39 cm · 224–710 g','wingspan': '80–95 cm',
        'fact': 'Its heart shaped facial disc acts as a parabolic reflector, giving it hearing so acute it can hunt in total darkness.',
        'silhouette': '🦉','color_band': 'wheat',},
    'CARMINE BEE-EATER': {
        'scientific': 'Merops nubicoides',
        'habitat': 'Cliffs & riverbanks of sub Saharan Africa',
        'diet': 'Flying insects especially bees, wasps and hornets',
        'size': '38 cm (incl. tail streamers) · 42–56 g','wingspan': '29–31 cm',
        'fact': 'Nests in vast colonies that can exceed 10000 pairs, excavating burrows up to 2m deep in sandy cliffs.',
        'silhouette': '🐦', 'color_band': 'crimson',},
    'DOWNY WOODPECKER': {
        'scientific': 'Dryobates pubescens',
        'habitat': 'Deciduous forests, orchards and suburban gardens across North America',
        'diet': 'Insects, beetle larvae, berries and seeds',
        'size': '14–17 cm · 21–28 g','wingspan': '25–31 cm',
        'fact': 'The smallest woodpecker in North America, yet its skull is specially reinforced to absorb up to 1200 g forces per peck.',
        'silhouette': '🐦','color_band': 'slategray',},
    'EMPEROR PENGUIN': {
        'scientific': 'Aptenodytes forsteri',
        'habitat': 'Antarctic pack ice and surrounding southern oceans',
        'diet': 'Fish, squid and krill','size': '100–130 cm · 22–45 kg','wingspan': '76–89 cm (flippers)',
        'fact': 'The only animal to breed during the Antarctic winter. Males fast for 65+ days while incubating a single egg.',
        'silhouette': '🐧','color_band': 'steelblue',},
    'FLAMINGO': {
        'scientific': 'Phoenicopterus roseus',
        'habitat': 'Shallow saline lakes, lagoons and estuaries worldwide',
        'diet': 'Algae, crustaceans and plankton filtered through its unique upside-down bill',
        'size': '110–150 cm · 2–4 kg','wingspan': '95–100 cm',
        'fact': 'Their iconic pink colouration comes entirely from carotenoid pigments in their food. Captive flamingos fed plain diets turn white.',
        'silhouette': '🦩','color_band': 'hotpink',},
}
CLASS_NAMES = list(BIRD_INFO.keys())

#model 
model_path = "project_model_dlor2.keras"
@st.cache_resource
def load_bird_model():
    if not os.path.exists(model_path):
        st.error(f"Model not found: {model_path}")
        return None
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
model = load_bird_model()

def preprocess(img):
    img = img.resize((224, 224))
    arr = tf.keras.utils.img_to_array(img)
    return np.expand_dims(arr, axis=0)

#sidebar
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 0.8rem; border-bottom:2px solid burlywood; margin-bottom:1rem;'>
        <p style='font-family:Crimson Pro,Georgia,serif; font-size:0.7rem; color:peru;
                  letter-spacing:0.25em; text-transform:uppercase; margin:0;'>Guide to Birds</p>
        <h2 style='font-family:Crimson Pro,Georgia,serif; color:maroon; margin:0.2rem 0 0;
                   font-size:1.5rem; font-style:italic;'>Bird Enthusiast</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style='font-family:JetBrains Mono,monospace; font-size:0.68rem; color:peru;
              letter-spacing:0.15em; text-transform:uppercase; margin-bottom:0.6rem;'>
    Identifiable Birds in this Website
    </p>
    """, unsafe_allow_html=True)

    for species, info in BIRD_INFO.items():
        st.markdown(f"""
        <div style='display:flex; align-items:flex-start; gap:0.5rem; padding:0.5rem 0;
                    border-bottom:1px solid rgba(139,90,43,0.12);'>
            <span style='font-size:1.3rem; line-height:1;'>{info['silhouette']}</span>
            <div style='flex:1;'>
                <p style='font-family:Crimson Pro,Georgia,serif; font-size:1rem;
                          color:maroon; margin:0; font-weight:600; line-height:1.2;'>{species.title()}</p>
                <p style='font-family:JetBrains Mono,monospace; font-size:0.75rem;
                          color:peru; margin:0; font-style:italic;'>{info['scientific']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:1.2rem; padding:0.9rem; background:linen;
                border:1px solid burlywood; border-radius:3px;'>
        <p style='font-family:Crimson Pro,Georgia,serif; font-size:0.8rem; color:saddlebrown;
                  margin:0; font-style:italic; line-height:1.6;'>
        "In every walk with Nature, one receives far more than he seeks."
        </p>
        <p style='font-family:JetBrains Mono,monospace; font-size:0.65rem; color:peru;
                  margin:0.4rem 0 0; letter-spacing:0.05em;'>— John Muir</p>
    </div>
    """,unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:1rem;'>
        <p style='font-family:JetBrains Mono,monospace; font-size:0.65rem; color:burlywood;
                  letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.4rem;'>Tips for best AI identification results</p>
        <p style='font-family:Libre Baskerville,serif; font-size:0.78rem; color:saddlebrown; line-height:1.7;'>
        • Clear and unobstructed view of the bird<br>
        • Natural daylight preferred<br>
        • Single subject per image<br>
        • JPG or PNG format
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style='border-bottom:3px double saddlebrown; padding:2rem 0 1.5rem; margin-bottom:2rem;'>
    <div style='display:flex; align-items:flex-end; justify-content:space-between; flex-wrap:wrap; gap:1rem;'>
        <div>
            <h1 style='font-family:Crimson Pro,Georgia,serif; font-size:3.2rem; color:maroon;
                       margin:0; line-height:1; letter-spacing:-0.02em;'>
                Bird Species Classifier
            </h1>
            <p style='font-family:Libre Baskerville,Georgia,serif; font-size:1.05rem;
                      color:saddlebrown; margin:0.4rem 0 0; font-style:italic;'>
                Identify bird species using AI
            </p>
        </div>
        <div style='text-align:right;'>
            <p style='font-family:JetBrains Mono,monospace; font-size:3rem; color:burlywood;a
                      margin:0; line-height:1; opacity:0.6;'>🪶</p>
            <p style='font-family:JetBrains Mono,monospace; font-size:0.65rem; color:peru;
                      letter-spacing:0.1em; margin:0.2rem 0 0;'>SUPPORTS 6 SPECIES</p>
        </div>
    </div>
</div>
""",unsafe_allow_html=True)

#demo images
demo_dir = "demo_images"
st.markdown("""
<p style='font-family:JetBrains Mono,monospace; font-size:0.68rem; color:peru;
          letter-spacing:0.2em; text-transform:uppercase; margin-bottom:0.8rem;'>
Choose a demo pic to instantly identify the species
</p>
""", unsafe_allow_html=True)

def square_crop(img, size=240):
    w, h=img.size
    side= min(w, h)
    left= (w - side)// 2
    top =(h - side)//2
    img= img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.LANCZOS)

if os.path.exists(demo_dir):
    demo_files= [f for f in os.listdir(demo_dir) if f.lower().endswith((".jpg", ".png"))]
    if demo_files:
        gcols=st.columns(min(len(demo_files), 6), gap="small")
        for i, file in enumerate(demo_files[:6]):
            img_path=os.path.join(demo_dir, file)
            demo_img= Image.open(img_path).convert("RGB")
            tile_img =square_crop(demo_img)
            label = file.split('.')[0].replace('_', ' ').title()
            with gcols[i % 6]:
                st.image(tile_img, use_container_width=True)
                if st.button(label, key=f"demo_{file}"):
                    buf = BytesIO()
                    demo_img.save(buf, format='PNG')
                    buf.seek(0)
                    st.session_state["bird_image"] = Image.open(buf).convert("RGB")
                    st.session_state["bird_source"] = label
else:
    st.markdown("""
    <div style='border:1px dashed burlywood; border-radius:3px; padding:1.2rem;
                background:antiquewhite; text-align:center;'>
        <p style='font-family:Crimson Pro,Georgia,serif; color:saddlebrown; font-style:italic; margin:0;'>
        Create a <code>demo_images/</code> folder with sample bird photographs to populate the gallery.
        </p>
    </div>
    """, unsafe_allow_html=True)
st.markdown("<hr style='margin:1.5rem 0;'>", unsafe_allow_html=True)

#upload section
if "bird_image" not in st.session_state:
    st.session_state["bird_image"]= None
if "bird_source" not in st.session_state:
    st.session_state["bird_source"]= None
if "bird_result" not in st.session_state:
    st.session_state["bird_result"]= None

st.markdown("""
<p style='font-family:JetBrains Mono,monospace; font-size:0.82rem; color:peru;
          letter-spacing:0.2em; text-transform:uppercase; margin-bottom:0.8rem;'>
Upload picture of the bird
</p>
""", unsafe_allow_html=True)
up_col, action_col = st.columns([2, 1], gap="large")
with up_col:
    uploaded = st.file_uploader(
        "Upload field photograph",
        type=["jpg","jpeg","png"],
        label_visibility="collapsed"
    )
    if uploaded:
        if st.session_state["bird_source"]!= uploaded.name:
            st.session_state["bird_image"] = Image.open(uploaded).convert("RGB")
            st.session_state["bird_source"]=uploaded.name
            st.session_state["bird_result"] = None 
            st.rerun()

image = st.session_state["bird_image"]
with action_col:
    if image:
        st.markdown("""
        <p style='font-family:JetBrains Mono,monospace; font-size:0.78rem; color:peru;
                  letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.5rem;'>
        Preview Picture
        </p>
        """, unsafe_allow_html=True)
        thumb = image.copy()
        thumb.thumbnail((200, 200))
        st.image(thumb, use_container_width=True)
        rem_col, _ = st.columns([1, 1])
        with rem_col:
            if st.button("✕ Clear", key="clear_btn"):
                st.session_state["bird_image"]= None
                st.session_state["bird_source"]=None
                st.session_state["bird_result"]= None  
                st.rerun()

st.markdown("<br>",unsafe_allow_html=True)

#classify button 
if image and model:
    if st.button("Identify Bird", key="classify_btn"):
        with st.spinner("Checking the Beak and Feathers..."):
            t0 = time.time()
            tensor = preprocess(image)
            preds  = model.predict(tensor, verbose=0)[0]
            elapsed = time.time() - t0
        st.session_state["bird_result"] = {
            "preds": preds.tolist(),"elapsed":elapsed,
            "source":st.session_state.get("bird_source","uploaded image"),
        }
        st.rerun()
elif image and not model:
    st.warning("Model not loaded, please ensure the model file is present.")

#results display
result= st.session_state.get("bird_result")
if result and image:
    preds = np.array(result["preds"])
    elapsed= result["elapsed"]
    idx=int(np.argmax(preds))
    species=CLASS_NAMES[idx]
    confidence= float(preds[idx])
    info =BIRD_INFO[species]
    color_accent= info['color_band']

    if confidence>0.82:
        tier, tier_color, tier_label = "●", "forestgreen", "High certainty"
    elif confidence > 0.60:
        tier,tier_color, tier_label = "◐", "darkorange", "Moderate certainty"
    else:
        tier, tier_color, tier_label = "○", "crimson", "Low certainty (verify manually)"
    st.markdown("<hr style='margin:2rem 0 1.5rem;'>", unsafe_allow_html=True)
    rec_left, rec_right = st.columns([1.2, 1], gap="large")
    with rec_left:
        st.image(image, use_container_width=True,
                 caption=f"bird: {result['source']}")
    with rec_right:
        st.markdown(f"""
        <div style='border:2px solid saddlebrown; border-radius:3px;
                    background:antiquewhite; overflow:hidden;'>
            <div style='height:6px; background:{color_accent}; width:100%;'></div>
            <div style='padding:1.5rem;'>
                <p style='font-family:JetBrains Mono,monospace; font-size:0.78rem; color:peru;
                          letter-spacing:0.2em; text-transform:uppercase; margin:0 0 0.3rem;'>Bird identified</p>
                <h2 style='font-family:Crimson Pro,Georgia,serif; font-size:2.4rem; color:maroon;
                           margin:0; line-height:1.1;'>{species.title()}</h2>
                <p style='font-family:JetBrains Mono,monospace; font-size:0.95rem; color:peru;
                          font-style:italic; margin:0.2rem 0 1rem;'>{info['scientific']}</p>
                <div style='display:flex; align-items:center; gap:0.8rem; margin-bottom:1rem;'>
                    <span style='font-size:1.6rem; color:{tier_color};'>{tier}</span>
                    <div>
                        <p style='font-family:JetBrains Mono,monospace; font-size:1.15rem;
                                  color:darkslategray; margin:0; font-weight:500;'>{confidence*100:.1f}% confidence</p>
                        <p style='font-family:JetBrains Mono,monospace; font-size:0.82rem;
                                  color:{tier_color}; margin:0;'>{tier_label}</p>
                    </div>
                </div>
                <div style='background:wheat; border-radius:2px; height:8px; margin-bottom:1.2rem;'>
                    <div style='width:{int(confidence*100)}%; height:100%;
                                background:{color_accent}; border-radius:2px;'></div>
                </div>
                <table style='width:100%; border-collapse:collapse; font-size:0.95rem;'>
                    <tr style='border-bottom:1px solid burlywood;'>
                        <td style='padding:0.45rem 0; color:peru; font-family:JetBrains Mono,monospace;
                                   font-size:0.78rem; text-transform:uppercase; width:38%;'>Size</td>
                        <td style='padding:0.45rem 0; color:darkslategray; font-family:JetBrains Mono,monospace;
                                   font-size:0.92rem;'>{info['size']}</td>
                    </tr>
                    <tr>
                        <td style='padding:0.45rem 0; color:peru; font-family:JetBrains Mono,monospace;
                                   font-size:0.78rem; text-transform:uppercase;'>Wingspan</td>
                        <td style='padding:0.45rem 0; color:darkslategray; font-family:JetBrains Mono,monospace;
                                   font-size:0.92rem;'>{info['wingspan']}</td>
                    </tr>
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style='margin-top:0.8rem; border-left:3px solid {color_accent};
                    padding:0.9rem 1rem; background:linen; border-radius:0 3px 3px 0;'>
            <p style='font-family:JetBrains Mono,monospace; font-size:0.78rem; color:peru;
                      letter-spacing:0.15em; text-transform:uppercase; margin:0 0 0.5rem;'>Fun fact</p>
            <p style='font-family:Libre Baskerville,serif; font-size:0.98rem;
                      color:darkslategray; margin:0; line-height:1.7; font-style:italic;'>{info['fact']}</p>
        </div>
        """, unsafe_allow_html=True)

    #habiteat and diet
    st.markdown("<br>", unsafe_allow_html=True)
    h_col,d_col = st.columns(2, gap="large")
    with h_col:
        st.markdown(f"""
        <div style='border:1px solid burlywood; border-radius:3px; padding:1.1rem; background:antiquewhite;'>
            <p style='font-family:JetBrains Mono,monospace; font-size:0.78rem; color:peru;
                      letter-spacing:0.15em; text-transform:uppercase; margin:0 0 0.5rem;'>Habitat</p>
            <p style='font-family:Libre Baskerville,serif; font-size:1rem;
                      color:darkslategray; margin:0; line-height:1.65;'>{info['habitat']}</p>
        </div>
        """, unsafe_allow_html=True)
    with d_col:
        st.markdown(f"""
        <div style='border:1px solid burlywood; border-radius:3px; padding:1.1rem; background:antiquewhite;'>
            <p style='font-family:JetBrains Mono,monospace; font-size:0.78rem; color:peru;
                      letter-spacing:0.15em; text-transform:uppercase; margin:0 0 0.5rem;'>Diet</p>
            <p style='font-family:Libre Baskerville,serif; font-size:1rem;
                      color:darkslategray; margin:0; line-height:1.65;'>{info['diet']}</p>
        </div>
        """, unsafe_allow_html=True)

    #probability breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Probablity of the Birds", expanded=False):
        st.markdown("""
        <p style='font-family:JetBrains Mono,monospace; font-size:0.85rem; color:peru; margin-bottom:1rem;'>
        Model Prediction Percentages
        </p>
        """, unsafe_allow_html=True)
        for rank, i in enumerate(np.argsort(preds)[::-1]):
            sp_name= CLASS_NAMES[i]
            sp_prob=float(preds[i])
            sp_info=BIRD_INFO[sp_name]
            bar_pct= int(sp_prob * 100)
            is_top = (i == idx)
            bar_col= sp_info['color_band'] if is_top else "burlywood"
            row_bg ="linen" if is_top else "transparent"
            border= "saddlebrown" if is_top else "transparent"
            name_color = "maroon" if is_top else "darkslategray"
            name_weight= "700" if is_top else "400"
            pct_weight= "600" if is_top else "400"
            st.markdown(f"""
            <div style='padding:0.6rem 0.8rem; margin-bottom:0.4rem;
                        background:{row_bg}; border-radius:3px; border:1px solid {border};'>
                <div style='display:flex; justify-content:space-between; align-items:baseline; margin-bottom:0.35rem;'>
                    <div style='display:flex; align-items:center; gap:0.5rem;'>
                        <span style='font-family:JetBrains Mono,monospace; font-size:0.75rem; color:peru;'>#{rank+1}</span>
                        <span style='font-family:Crimson Pro,Georgia,serif; font-size:1.05rem;
                                     color:{name_color}; font-weight:{name_weight};'>{sp_name.title()}</span>
                        <span style='font-family:JetBrains Mono,monospace; font-size:0.75rem;
                                     color:peru; font-style:italic;'>{sp_info['scientific']}</span>
                    </div>
                    <span style='font-family:JetBrains Mono,monospace; font-size:0.95rem;
                                 color:{name_color}; font-weight:{pct_weight};'>{sp_prob*100:.2f}%</span>
                </div>
                <div style='height:6px; background:wheat; border-radius:99px;'>
                    <div style='width:{bar_pct}%; height:100%; background:{bar_col}; border-radius:99px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f"""
        <p style='font-family:JetBrains Mono,monospace; font-size:0.75rem; color:burlywood; margin-top:0.8rem;'>
        Time taken: {elapsed*1000:.0f} ms</p>
        """, unsafe_allow_html=True)
elif not image:
    st.markdown("""
    <div style='border:2px dashed burlywood; border-radius:4px; padding:3rem 2rem;
                background:antiquewhite; text-align:center; margin:1rem 0;'>
        <p style='font-size:3rem; margin:0 0 0.8rem;'>🪶</p>
        <h3 style='font-family:Crimson Pro,Georgia,serif; color:maroon; font-size:1.7rem;
                   margin:0 0 0.5rem; font-style:italic;'>No picture uploaded</h3>
        <p style='font-family:Libre Baskerville,serif; color:darkslategray; max-width:400px;
                  margin:0 auto; line-height:1.7; font-size:1rem;'>
            Upload a demo picture from above or upload your own picture of a bird to identify it with AI.
        </p>
    </div>
    """, unsafe_allow_html=True)