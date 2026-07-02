import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
import io, base64, json
from IPython.display import HTML


def fig_to_b64(im):
    fig, ax = plt.subplots()
    ax.imshow(im)            # add cmap='gray' if these are single-channel
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()



def keyswap_imgs(images, keys=['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p']):

    mapping = {k: fig_to_b64(im) for k, im in zip(keys, images)}

    html = f"""
    <div tabindex="0" id="viewer" style="outline:none;">
    <img id="display" src="data:image/png;base64,{mapping[keys[0]]}" style="max-width:100%;">
    <p>{', '.join(keys)}</p>
    </div>
    <script>
    const imgs = {json.dumps(mapping)};
    const viewer = document.getElementById('viewer');
    viewer.focus();
    viewer.addEventListener('keydown', (e) => {{
        if (e.key in imgs) {{
        document.getElementById('display').src = 'data:image/png;base64,' + imgs[e.key];
        }}
    }});
    </script>
    """
    HTML(html)