# Tennis Autodistillation

Tennis Autodistillation is a project aimed at applying autodistillation techniques to tennis video analysis.

This project creates foundation models to generate datasets to train the following models:
- Frame classification
- Tennis player detector
- Court keypoints detector

Watch our (demo video)[https://drive.google.com/file/d/1jyFV8Z-Xo9iQFWdJeeerkqdbP0DuoRWq/view?usp=sharing]
## Installation

Create an environment, so that you ...
```bash
conda create -n tennis-autodistill python=3.13
conda activate tennis-autodistill
```

To install the project, clone the repository and install the dependencies:
```bash
git clone https://github.com/yourusername/tennis-autodistill.git
cd tennis-autodistillation
pip install -r requirements.txt
pip install -e .
```

## Usage

To use the project, follow the notebook `tennis_autodistill.ipynb`.

I highly recommend to watch the instruction videos first. They will help you understand the project and the decisions behind the code.
- (The pipeline and overview of the project)[https://drive.google.com/file/d/1EftccAKo-N-nUBJ5qratIvBshGeHQcJX/view?usp=sharing]
- (Dataset creation)[https://drive.google.com/file/d/176dd0X9VBY4WWYObcxooYH8lj-336EoV/view?usp=sharing]
- (Estimating court points)[https://drive.google.com/file/d/1apjGZCoc_KYSnWwLeMuUVPVg7nsdCmp-/view?usp=sharing]




https://drive.google.com/file/d/1apjGZCoc_KYSnWwLeMuUVPVg7nsdCmp-/view?usp=sharing

output-short-rally.mp4
https://drive.google.com/file/d/1jyFV8Z-Xo9iQFWdJeeerkqdbP0DuoRWq/view?usp=sharing




## Court minimap
International Tennis Federation (ITF) rules of tennis:
https://www.itftennis.com/media/7221/2025-rules-of-tennis-english.pdf

https://www.harrodsport.com/advice-and-guides/tennis-court-dimensions#:~:text=A%20tennis%20court%20is%2078ft,6.4m)%20from%20the%20net.


## Acknowledgments

- List any resources, libraries, or individuals you want to acknowledge.
