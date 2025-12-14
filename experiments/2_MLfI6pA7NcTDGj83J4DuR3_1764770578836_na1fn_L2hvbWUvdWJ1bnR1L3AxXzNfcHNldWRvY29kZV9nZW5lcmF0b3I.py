import os

def generate_latex_pseudocode():
    """
    生成包含5个算法伪代码的LaTeX文件。
    
    该函数定义了5个常用算法（归并排序、Dijkstra算法、K-Means聚类、反向传播、A*搜索）
    的LaTeX伪代码，并使用algorithm2e宏包将其封装在一个完整的LaTeX文档中。
    
    输出文件名为 'algorithms_pseudocode.tex'。
    """
    
    # 使用algorithm2e宏包来编写伪代码
    latex_header = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[linesnumbered,ruled,vlined]{algorithm2e}
\usepackage{geometry}
\geometry{a4paper, margin=1in}

\title{Five Algorithms Pseudocode for P1-3}
\author{Manus AI}
\date{\today}

\begin{document}
\maketitle

\section{Algorithms Pseudocode}

"""

    # 1. Merge Sort (归并排序)
    merge_sort_pseudocode = r"""
\subsection{Merge Sort Algorithm}
\begin{algorithm}[H]
\SetAlgoLined
\KwData{An array $A$ of $n$ elements}
\KwResult{A sorted array $A$}
\SetKwFunction{FMergeSort}{MergeSort}
\SetKwFunction{FMerge}{Merge}
\SetKwInOut{Input}{Input}\SetKwInOut{Output}{Output}

\Input{Array $A$, index $p$, index $r$}
\Output{Sorted subarray $A[p..r]$}

\Fn{\FMergeSort{$A, p, r$}}{
    \If{$p < r$}{
        $q \leftarrow \lfloor (p+r)/2 \rfloor$\;
        \FMergeSort{$A, p, q$}\;
        \FMergeSort{$A, q+1, r$}\;
        \FMerge{$A, p, q, r$}\;
    }
}

\Input{Array $A$, index $p$, index $q$, index $r$}
\Output{Merged and sorted subarray $A[p..r]$}

\Fn{\FMerge{$A, p, q, r$}}{
    $n_1 \leftarrow q - p + 1$\;
    $n_2 \leftarrow r - q$\;
    Create arrays $L[1..n_1+1]$ and $R[1..n_2+1]$\;
    \For{$i \leftarrow 1$ \KwTo $n_1$}{
        $L[i] \leftarrow A[p + i - 1]$\;
    }
    \For{$j \leftarrow 1$ \KwTo $n_2$}{
        $R[j] \leftarrow A[q + j]$\;
    }
    $L[n_1 + 1] \leftarrow \infty$\;
    $R[n_2 + 1] \leftarrow \infty$\;
    $i \leftarrow 1$\;
    $j \leftarrow 1$\;
    \For{$k \leftarrow p$ \KwTo $r$}{
        \If{$L[i] \le R[j]$}{
            $A[k] \leftarrow L[i]$\;
            $i \leftarrow i + 1$\;
        }
        \Else{
            $A[k] \leftarrow R[j]$\;
            $j \leftarrow j + 1$\;
        }
    }
}
\caption{Merge Sort and Merge}
\end{algorithm}
"""

    # 2. Dijkstra's Algorithm (Dijkstra算法)
    dijkstra_pseudocode = r"""
\subsection{Dijkstra's Algorithm}
\begin{algorithm}[H]
\SetAlgoLined
\KwData{Graph $G=(V, E)$, source vertex $s$}
\KwResult{Shortest path distance $d[v]$ from $s$ to all $v \in V$}
\SetKwFunction{FInit}{Initialize-Single-Source}
\SetKwFunction{FExtractMin}{Extract-Min}
\SetKwFunction{FRelax}{Relax}

\Fn{\FInit{$G, s$}}{
    \For{each vertex $v \in V$}{
        $d[v] \leftarrow \infty$\;
        $\pi[v] \leftarrow \text{NIL}$\;
    }
    $d[s] \leftarrow 0$\;
}

\Fn{Dijkstra{$G, s$}}{
    \FInit{$G, s$}\;
    $S \leftarrow \emptyset$ \tcp*{Set of vertices whose final shortest-path weight has been determined}
    $Q \leftarrow V$ \tcp*{Min-priority queue of vertices}
    \While{$Q \neq \emptyset$}{
        $u \leftarrow \FExtractMin{$Q$}\;
        $S \leftarrow S \cup \{u\}$\;
        \For{each vertex $v \in \text{Adj}[u]$}{
            \Fn{\FRelax{$u, v, w$}}{
                \If{$d[v] > d[u] + w(u, v)$}{
                    $d[v] \leftarrow d[u] + w(u, v)$\;
                    $\pi[v] \leftarrow u$\;
                }
            }
            \FRelax{$u, v, w$}\;
        }
    }
}
\caption{Dijkstra's Algorithm}
\end{algorithm}
"""

    # 3. K-Means Clustering (K-Means聚类)
    kmeans_pseudocode = r"""
\subsection{K-Means Clustering Algorithm}
\begin{algorithm}[H]
\SetAlgoLined
\KwData{Dataset $X = \{x_1, x_2, \dots, x_n\}$, Number of clusters $K$}
\KwResult{Set of $K$ clusters, Cluster centroids $\mu_1, \dots, \mu_K$}

\Fn{KMeans{$X, K$}}{
    Initialize $K$ centroids $\mu_1, \dots, \mu_K$ randomly from $X$\;
    \Repeat{centroids do not change}{
        \For{each data point $x_i \in X$}{
            $c_i \leftarrow \arg\min_j \|x_i - \mu_j\|^2$ \tcp*{Assignment step: Assign $x_i$ to the closest centroid}
        }
        \For{$j \leftarrow 1$ \KwTo $K$}{
            $\mu_j \leftarrow \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i$ \tcp*{Update step: Recalculate centroid $\mu_j$}
        }
    }
    \Return $\mu_1, \dots, \mu_K$ and clusters $C_1, \dots, C_K$\;
}
\caption{K-Means Clustering}
\end{algorithm}
"""

    # 4. Backpropagation (反向传播)
    backpropagation_pseudocode = r"""
\subsection{Backpropagation Algorithm}
\begin{algorithm}[H]
\SetAlgoLined
\KwData{Training set $D$, Learning rate $\eta$, Network architecture}
\KwResult{Trained network weights}
\SetKwFunction{FForward}{Forward-Pass}
\SetKwFunction{FBackward}{Backward-Pass}

\Fn{Backpropagation{$D, \eta$}}{
    Initialize all weights and biases randomly\;
    \Repeat{stopping criterion is met}{
        \For{each training example $(x, y) \in D$}{
            \tcp{Forward Pass}
            $a^{(0)} \leftarrow x$\;
            \For{$l \leftarrow 1$ \KwTo $L$}{
                $z^{(l)} \leftarrow W^{(l)} a^{(l-1)} + b^{(l)}$\;
                $a^{(l)} \leftarrow \sigma(z^{(l)})$ \tcp*{Activation function $\sigma$}
            }
            
            \tcp{Backward Pass}
            $\delta^{(L)} \leftarrow (a^{(L)} - y) \odot \sigma'(z^{(L)})$ \tcp*{Output layer error}
            \For{$l \leftarrow L-1$ \KwTo $1$}{
                $\delta^{(l)} \leftarrow ((W^{(l+1)})^T \delta^{(l+1)}) \odot \sigma'(z^{(l)})$ \tcp*{Hidden layer error}
            }
            
            \tcp{Weight Update}
            \For{$l \leftarrow 1$ \KwTo $L$}{
                $W^{(l)} \leftarrow W^{(l)} - \eta \delta^{(l)} (a^{(l-1)})^T$\;
                $b^{(l)} \leftarrow b^{(l)} - \eta \delta^{(l)}$\;
            }
        }
    }
    \Return $W, b$\;
}
\caption{Backpropagation for Neural Networks}
\end{algorithm}
"""

    # 5. A* Search Algorithm (A*搜索算法)
    astar_pseudocode = r"""
\subsection{A* Search Algorithm}
\begin{algorithm}[H]
\SetAlgoLined
\KwData{Start node $start$, Goal node $goal$, Heuristic function $h(n)$}
\KwResult{Path from $start$ to $goal$}
\SetKwFunction{FReconstruct}{Reconstruct\_Path}

\Fn{AStarSearch{$start, goal$}}{
    $openSet \leftarrow \{start\}$ \tcp*{Nodes to be evaluated}
    $cameFrom \leftarrow \text{empty map}$ \tcp*{For reconstructing the path}
    
    $gScore[start] \leftarrow 0$ \tcp*{Cost from start along best known path}
    $fScore[start] \leftarrow gScore[start] + h(start)$ \tcp*{Estimated total cost}
    
    \While{$openSet$ is not empty}{
        $current \leftarrow \text{node in } openSet \text{ having the lowest } fScore$\;
        \If{$current = goal$}{
            \Return \FReconstruct{$cameFrom, current$}\;
        }
        
        $openSet \leftarrow openSet \setminus \{current\}$\;
        \For{each neighbor $neighbor$ of $current$}{
            $tentative\_gScore \leftarrow gScore[current] + \text{dist}(current, neighbor)$\;
            \If{$tentative\_gScore < gScore[neighbor]$}{
                $cameFrom[neighbor] \leftarrow current$\;
                $gScore[neighbor] \leftarrow tentative\_gScore$\;
                $fScore[neighbor] \leftarrow gScore[neighbor] + h(neighbor)$\;
                \If{$neighbor \notin openSet$}{
                    $openSet \leftarrow openSet \cup \{neighbor\}$\;
                }
            }
        }
    }
    \Return \text{Failure}\;
}
\caption{A* Search Algorithm}
\end{algorithm}
"""

    latex_footer = r"""
\end{document}
"""

    # 组合所有部分
    full_latex_document = (
        latex_header +
        merge_sort_pseudocode +
        dijkstra_pseudocode +
        kmeans_pseudocode +
        backpropagation_pseudocode +
        astar_pseudocode +
        latex_footer
    )

    output_file = "algorithms_pseudocode.tex"
    
    # 写入文件
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_latex_document)
        print(f"Successfully generated LaTeX pseudocode file: {output_file}")
        return True, output_file
    except Exception as e:
        print(f"Error writing file: {e}")
        return False, str(e)

# 测试验证部分
def test_generation():
    """
    执行生成函数并验证文件是否成功创建。
    """
    print("Starting test verification...")
    success, result = generate_latex_pseudocode()
    
    if success:
        # 验证文件是否存在
        if os.path.exists(result):
            print(f"Test Passed: File '{result}' exists.")
            # 验证文件内容是否非空
            if os.path.getsize(result) > 1000: # 伪代码文件应该大于1KB
                print(f"Test Passed: File size is {os.path.getsize(result)} bytes (non-empty).")
                return True, f"File '{result}' successfully generated and is non-empty."
            else:
                print(f"Test Failed: File size is too small ({os.path.getsize(result)} bytes).")
                return False, f"File size is too small ({os.path.getsize(result)} bytes)."
        else:
            print(f"Test Failed: File '{result}' does not exist.")
            return False, f"File '{result}' does not exist."
    else:
        print(f"Test Failed: Generation failed with error: {result}")
        return False, f"Generation failed with error: {result}"

if __name__ == "__main__":
    # 运行测试
    test_passed, test_message = test_generation()
    print("\n--- Final Test Result ---")
    print(f"Status: {'SUCCESS' if test_passed else 'FAILURE'}")
    print(f"Message: {test_message}")
    
    # 如果需要，可以尝试编译LaTeX文件以进行更彻底的验证
    # 但由于沙箱环境可能没有pdflatex，我们只验证文件生成和内容。
    # 如果需要编译，可以使用 shell 工具执行 pdflatex algorithms_pseudocode.tex
    
# 预期输出文件路径
CODE_FILE_PATH = os.path.abspath("p1_3_pseudocode_generator.py")
OUTPUT_FILE_PATH = os.path.abspath("algorithms_pseudocode.tex")
# print(f"Code File Path: {CODE_FILE_PATH}")
# print(f"Output File Path: {OUTPUT_FILE_PATH}")
