# Text Style Transfer RNN

🎭 *"To train, or not to train— that is the question."*

## How it Works

This project leverages Recurrent Neural Networks (RNNs) and their advanced variants. Let's break down how these architectures function:

### RNN (Recurrent Neural Network)

Think of RNNs as neural networks with a memory. Unlike regular networks that process data in isolation, RNNs maintain a "hidden state" that carries information from previous inputs. At each step, the network:

1. Takes in new input
2. Combines it with the previous hidden state
3. Produces an output and updates the hidden state

However, basic RNNs struggle with long-term dependencies due to vanishing/exploding gradients. Enter LSTMs and GRUs!

### LSTM (Long Short-Term Memory)

LSTMs are like RNNs with a sophisticated memory management system. They have three gates:

- Forget gate: Decides what to throw away from cell state
- Input gate: Decides what new information to store
- Output gate: Decides what parts of the cell state to output

This architecture helps maintain long-term dependencies and manages information flow more effectively.

### GRU (Gated Recurrent Unit)

GRUs are the streamlined cousin of LSTMs. They merge the forget and input gates into a single "update gate" and combine cell state with hidden state. This makes them:

- Computationally more efficient
- Often just as effective as LSTMs
- Easier to train due to fewer parameters

Each architecture has its strengths, but they all share the fundamental ability to process sequential data while maintaining context - perfect for text generation!


## Installation

1) Clone the repository:

    ```bash
    git clone https://github.com/johnhangen/TextStyleTransferRNN
    cd TextStyleTransferRNN
    ```

2) Create and activate a virtual environment (optional but recommended):

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3) Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4) Make sure to download the datasets and place it in the data/ folder.

## Usage

To use the model you can run the main.py file. This will train the model on the datasets and then generate text based on the seed you provide:

```bash
python main.py
```

## Data

For this project, I used the tiny Shakespeare dataset, Wikipedia, Linux Kernel, and a sample of my notes from school. For each of these datasets I trained a separate RNN. Below are samples are the results of using one random char as the seed for each of the datasets.

### Shakespeare

```txt
What is't sinched us you make budget-pear live him.

LARTIUS:
They from this blood
Bf our father's mood
At Grecian swear'd and endure
Was prick in thing this wetward then rejourn a cuprocenable confouths, if
Their blove answer'd!

COMINIUS:
Phostur, is bale
prie last, where that so stand same make gods.
Wherein we have but proceed
Be as mine home,
And carry things go valiak,
Than Hector'd.

COMINIUS:
Go whall,--

MARCIUS:
I am glad to easing good
The lamb. You
had suckled pace it brief Rome done in where is the cheacher,
Whiles for you,
Which? have set then; or some on, we
The brind their blood,
When she day high,
But him run incounsellor consper it brings a crack
Prad they last of thigh
That a guin'd thee. Hoo lant newing strange.

MARCIUS:
Comuniertolign on Marcius.

COMINIUS:
He crack, and held; I lefle son makes.

First Citizen:
You are fought to time and Caickly.

Second Senator:
How the clamon fame me, and go with us.

SICINIUS:
Look, lay as you you have lamb did be an of pride:
```

### Wikipedia

```txt
to be stocked or by accessive interests for the prophets, each decades, such as complete resource for all basis, for example, one of the [[WCBS]]; if they found an entire condition of the order, but learning puth of the populationâ€”whof the jurisdiction, considerably as in the Aston Menci.Dasaromatic County includes. And nearby was gained for around merged. This results should not be considered to be more explained in the universe.&quot;
```

### Linux Kernel

```c
VE_REG_INIT;a

        update_spin_unlock(&q->syst));
        for (rcu_ops->event_chunk_files);

        __read(" use each itsize sohe a CPUs.h>
#include <linux/post, " by the task the { nr_mutex);

        if (!rb) {
                event->context_event_buf, gcov_start)
                        device = 0;
        leget = h1);
}
void value;

        could flags, 0);
                        task_struct start_avg();;
        stop_irqstamp(&parent_cpu_invoke) *tr, zown) {
                if (!PIGC_TASK,
                                                         flags)
         *
 * calls-ring doing tasks mutex a the task to based += new_rd, exit);
                new = current->thaw = bp
```

### School Notes

```txt
```
