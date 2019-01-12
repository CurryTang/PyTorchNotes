---


---

<h1 id="pytorch入门api汇总">Pytorch入门API汇总</h1>
<blockquote>
<p>基于官方教程<a href="https://pytorch.org/tutorials/beginner/nn_tutorial.html">original</a></p>
</blockquote>
<p>在这篇笔记里我们将使用不同层级的几种Pytorch的API来完成任务。</p>
<p>我们接下来提到的几种工具是在Pytorch中最常用的，使用他们能够提升你的工作效率，同时也是使用Pytorch的"正确打开方式"。</p>
<ul>
<li>nn Module</li>
<li>DataLoader</li>
<li>Dataset</li>
</ul>
<h2 id="获取数据集">获取数据集</h2>
<p>首先我们获取要使用的MNIST数据集，在此先不使用Pytorch的API.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">from</span> pathlib <span class="token keyword">import</span> Path
<span class="token keyword">import</span> requests

DATA_PATH <span class="token operator">=</span> Path<span class="token punctuation">(</span><span class="token string">"data"</span><span class="token punctuation">)</span>
PATH <span class="token operator">=</span> DATA_PATH <span class="token operator">/</span> <span class="token string">"mnist"</span>

PATH<span class="token punctuation">.</span>mkdir<span class="token punctuation">(</span>parents<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">,</span> exist_ok<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>

URL <span class="token operator">=</span> <span class="token string">"http://deeplearning.net/data/mnist/"</span>
FILENAME <span class="token operator">=</span> <span class="token string">"mnist.pkl.gz"</span>

<span class="token keyword">if</span> <span class="token operator">not</span> <span class="token punctuation">(</span>PATH <span class="token operator">/</span> FILENAME<span class="token punctuation">)</span><span class="token punctuation">.</span>exists<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
        content <span class="token operator">=</span> requests<span class="token punctuation">.</span>get<span class="token punctuation">(</span>URL <span class="token operator">+</span> FILENAME<span class="token punctuation">)</span><span class="token punctuation">.</span>content
        <span class="token punctuation">(</span>PATH <span class="token operator">/</span> FILENAME<span class="token punctuation">)</span><span class="token punctuation">.</span><span class="token builtin">open</span><span class="token punctuation">(</span><span class="token string">"wb"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>write<span class="token punctuation">(</span>content<span class="token punctuation">)</span>


<span class="token keyword">import</span> pickle
<span class="token keyword">import</span> gzip

<span class="token keyword">with</span> gzip<span class="token punctuation">.</span><span class="token builtin">open</span><span class="token punctuation">(</span>PATH <span class="token operator">/</span> FILENAME<span class="token punctuation">,</span> <span class="token string">"rb"</span><span class="token punctuation">)</span> <span class="token keyword">as</span> f<span class="token punctuation">:</span>
        <span class="token punctuation">(</span><span class="token punctuation">(</span>x_train<span class="token punctuation">,</span> y_train<span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token punctuation">(</span>x_valid<span class="token punctuation">,</span> y_valid<span class="token punctuation">)</span><span class="token punctuation">,</span> _<span class="token punctuation">)</span> <span class="token operator">=</span> pickle<span class="token punctuation">.</span>load<span class="token punctuation">(</span>f<span class="token punctuation">,</span> encoding<span class="token operator">=</span><span class="token string">"latin-1"</span><span class="token punctuation">)</span>

<span class="token keyword">import</span> torch

x_train<span class="token punctuation">,</span> y_train<span class="token punctuation">,</span> x_valid<span class="token punctuation">,</span> y_valid <span class="token operator">=</span> <span class="token builtin">map</span><span class="token punctuation">(</span>
    torch<span class="token punctuation">.</span>tensor<span class="token punctuation">,</span> <span class="token punctuation">(</span>x_train<span class="token punctuation">,</span> y_train<span class="token punctuation">,</span> x_valid<span class="token punctuation">,</span> y_valid<span class="token punctuation">)</span>
<span class="token punctuation">)</span>
</code></pre>
<h2 id="api-1-barebone-pytorch">API 1: Barebone Pytorch</h2>
<p>不使用高级API, 使用最底层的Pytorch进行实现:</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">import</span> math

weights <span class="token operator">=</span> torch<span class="token punctuation">.</span>randn<span class="token punctuation">(</span><span class="token number">784</span><span class="token punctuation">,</span> <span class="token number">10</span><span class="token punctuation">)</span> <span class="token operator">/</span> math<span class="token punctuation">.</span>sqrt<span class="token punctuation">(</span><span class="token number">784</span><span class="token punctuation">)</span> <span class="token comment"># Xavier init here</span>
weights<span class="token punctuation">.</span>requires_grad_<span class="token punctuation">(</span><span class="token punctuation">)</span>
bias <span class="token operator">=</span> torch<span class="token punctuation">.</span>zeros<span class="token punctuation">(</span><span class="token number">10</span><span class="token punctuation">,</span> requires_grad<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>

<span class="token keyword">def</span> <span class="token function">log_softmax</span><span class="token punctuation">(</span>x<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token keyword">return</span> x <span class="token operator">-</span> x<span class="token punctuation">.</span>exp<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span><span class="token builtin">sum</span><span class="token punctuation">(</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">.</span>log<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span>unsqueeze<span class="token punctuation">(</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">)</span>

<span class="token comment"># @就是dot product</span>
<span class="token keyword">def</span> <span class="token function">model</span><span class="token punctuation">(</span>xb<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token keyword">return</span> log_softmax<span class="token punctuation">(</span>xb @ weights <span class="token operator">+</span> bias<span class="token punctuation">)</span>

<span class="token comment"># 现在preds上也包含了有关gradient的信息</span>
bs <span class="token operator">=</span> <span class="token number">64</span>  <span class="token comment"># batch size</span>
xb <span class="token operator">=</span> x_train<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">:</span>bs<span class="token punctuation">]</span>  <span class="token comment"># a mini-batch from x</span>
preds <span class="token operator">=</span> model<span class="token punctuation">(</span>xb<span class="token punctuation">)</span>  <span class="token comment"># predictions</span>
preds<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">,</span> preds<span class="token punctuation">.</span>shape
<span class="token keyword">print</span><span class="token punctuation">(</span>preds<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">,</span> preds<span class="token punctuation">.</span>shape<span class="token punctuation">)</span>

<span class="token keyword">def</span> <span class="token function">nll</span><span class="token punctuation">(</span><span class="token builtin">input</span><span class="token punctuation">,</span> target<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token keyword">return</span> <span class="token operator">-</span><span class="token builtin">input</span><span class="token punctuation">[</span><span class="token builtin">range</span><span class="token punctuation">(</span>target<span class="token punctuation">.</span>shape<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">,</span> target<span class="token punctuation">]</span><span class="token punctuation">.</span>mean<span class="token punctuation">(</span><span class="token punctuation">)</span>

loss_func <span class="token operator">=</span> nll
yb <span class="token operator">=</span> y_train<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">:</span>bs<span class="token punctuation">]</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>loss_func<span class="token punctuation">(</span>preds<span class="token punctuation">,</span> yb<span class="token punctuation">)</span><span class="token punctuation">)</span>
<span class="token keyword">def</span> <span class="token function">accuracy</span><span class="token punctuation">(</span>out<span class="token punctuation">,</span> yb<span class="token punctuation">)</span><span class="token punctuation">:</span>
    preds <span class="token operator">=</span> torch<span class="token punctuation">.</span>argmax<span class="token punctuation">(</span>out<span class="token punctuation">,</span> dim<span class="token operator">=</span><span class="token number">1</span><span class="token punctuation">)</span>
    <span class="token keyword">return</span> <span class="token punctuation">(</span>preds <span class="token operator">==</span> yb<span class="token punctuation">)</span><span class="token punctuation">.</span><span class="token builtin">float</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span>mean<span class="token punctuation">(</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>accuracy<span class="token punctuation">(</span>preds<span class="token punctuation">,</span> yb<span class="token punctuation">)</span><span class="token punctuation">)</span>

<span class="token comment"># training</span>
lr <span class="token operator">=</span> <span class="token number">0.5</span>  <span class="token comment"># learning rate</span>
epochs <span class="token operator">=</span> <span class="token number">2</span>  <span class="token comment"># how many epochs to train for</span>

<span class="token keyword">for</span> epoch <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span>epochs<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token keyword">for</span> i <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span><span class="token punctuation">(</span>n <span class="token operator">-</span> <span class="token number">1</span><span class="token punctuation">)</span> <span class="token operator">//</span> bs <span class="token operator">+</span> <span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token comment">#         set_trace()</span>
        start_i <span class="token operator">=</span> i <span class="token operator">*</span> bs
        end_i <span class="token operator">=</span> start_i <span class="token operator">+</span> bs
        xb <span class="token operator">=</span> x_train<span class="token punctuation">[</span>start_i<span class="token punctuation">:</span>end_i<span class="token punctuation">]</span>
        yb <span class="token operator">=</span> y_train<span class="token punctuation">[</span>start_i<span class="token punctuation">:</span>end_i<span class="token punctuation">]</span>
        pred <span class="token operator">=</span> model<span class="token punctuation">(</span>xb<span class="token punctuation">)</span>
        loss <span class="token operator">=</span> loss_func<span class="token punctuation">(</span>pred<span class="token punctuation">,</span> yb<span class="token punctuation">)</span>

        loss<span class="token punctuation">.</span>backward<span class="token punctuation">(</span><span class="token punctuation">)</span>
        <span class="token keyword">with</span> torch<span class="token punctuation">.</span>no_grad<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
            weights <span class="token operator">-=</span> weights<span class="token punctuation">.</span>grad <span class="token operator">*</span> lr
            bias <span class="token operator">-=</span> bias<span class="token punctuation">.</span>grad <span class="token operator">*</span> lr
            weights<span class="token punctuation">.</span>grad<span class="token punctuation">.</span>zero_<span class="token punctuation">(</span><span class="token punctuation">)</span>
            bias<span class="token punctuation">.</span>grad<span class="token punctuation">.</span>zero_<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<h2 id="引入高级api">引入高级API</h2>
<h3 id="torch.nn.functional">torch.nn.functional</h3>
<p>这一块提供了一些有用的loss function,比如</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">import</span> torch<span class="token punctuation">.</span>nn<span class="token punctuation">.</span>functional <span class="token keyword">as</span> F
loss_func <span class="token operator">=</span> F<span class="token punctuation">.</span>cross_entropy
</code></pre>
<h3 id="nn.module">nn.Module</h3>
<p>我们只要让自己的模型继承自nn.Module便可以使用了(类似Keras)<br>
我们一般在__init__ 中定义好Layer,并在forward中进行实际的运算。</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">from</span> torch <span class="token keyword">import</span> nn

<span class="token keyword">class</span> <span class="token class-name">Mnist_Logistic</span><span class="token punctuation">(</span>nn<span class="token punctuation">.</span>Module<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token keyword">def</span> <span class="token function">__init__</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token builtin">super</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span>__init__<span class="token punctuation">(</span><span class="token punctuation">)</span>
        self<span class="token punctuation">.</span>weights <span class="token operator">=</span> nn<span class="token punctuation">.</span>Parameter<span class="token punctuation">(</span>torch<span class="token punctuation">.</span>randn<span class="token punctuation">(</span><span class="token number">784</span><span class="token punctuation">,</span> <span class="token number">10</span><span class="token punctuation">)</span> <span class="token operator">/</span> math<span class="token punctuation">.</span>sqrt<span class="token punctuation">(</span><span class="token number">784</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
        self<span class="token punctuation">.</span>bias <span class="token operator">=</span> nn<span class="token punctuation">.</span>Parameter<span class="token punctuation">(</span>torch<span class="token punctuation">.</span>zeros<span class="token punctuation">(</span><span class="token number">10</span><span class="token punctuation">)</span><span class="token punctuation">)</span>

    <span class="token keyword">def</span> <span class="token function">forward</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> xb<span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token keyword">return</span> xb @ self<span class="token punctuation">.</span>weights <span class="token operator">+</span> self<span class="token punctuation">.</span>bias
</code></pre>
<p>有了这样一个模型后我们在训练时可以用model.parameters()直接调取要更新的参数。</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">with</span> torch<span class="token punctuation">.</span>no_grad<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
                <span class="token keyword">for</span> p <span class="token keyword">in</span> model<span class="token punctuation">.</span>parameters<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
                    p <span class="token operator">-=</span> p<span class="token punctuation">.</span>grad <span class="token operator">*</span> lr
                model<span class="token punctuation">.</span>zero_grad<span class="token punctuation">(</span><span class="token punctuation">)</span>

</code></pre>
<h3 id="nn.linear-and-so-on">nn.Linear and so on</h3>
<p>对于每一个具体的layer，pytorch其实也已经实现好了，这里我们举一个Linear的例子，对于conv2d, conv3d等都大同小异。</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">class</span> <span class="token class-name">Mnist_Logistic</span><span class="token punctuation">(</span>nn<span class="token punctuation">.</span>Module<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token keyword">def</span> <span class="token function">__init__</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token builtin">super</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span>__init__<span class="token punctuation">(</span><span class="token punctuation">)</span>
        self<span class="token punctuation">.</span>lin <span class="token operator">=</span> nn<span class="token punctuation">.</span>Linear<span class="token punctuation">(</span><span class="token number">784</span><span class="token punctuation">,</span> <span class="token number">10</span><span class="token punctuation">)</span>

    <span class="token keyword">def</span> <span class="token function">forward</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> xb<span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token keyword">return</span> self<span class="token punctuation">.</span>lin<span class="token punctuation">(</span>xb<span class="token punctuation">)</span>
</code></pre>
<h3 id="optim">optim</h3>
<p>使用torch的optim接口可以简化training部分的代码</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">get_model</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
    model <span class="token operator">=</span> Mnist_Logistic<span class="token punctuation">(</span><span class="token punctuation">)</span>
    <span class="token keyword">return</span> model<span class="token punctuation">,</span> optim<span class="token punctuation">.</span>SGD<span class="token punctuation">(</span>model<span class="token punctuation">.</span>parameters<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> lr<span class="token operator">=</span>lr<span class="token punctuation">)</span>

model<span class="token punctuation">,</span> opt <span class="token operator">=</span> get_model<span class="token punctuation">(</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>loss_func<span class="token punctuation">(</span>model<span class="token punctuation">(</span>xb<span class="token punctuation">)</span><span class="token punctuation">,</span> yb<span class="token punctuation">)</span><span class="token punctuation">)</span>

<span class="token keyword">for</span> epoch <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span>epochs<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token keyword">for</span> i <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span><span class="token punctuation">(</span>n <span class="token operator">-</span> <span class="token number">1</span><span class="token punctuation">)</span> <span class="token operator">//</span> bs <span class="token operator">+</span> <span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
        start_i <span class="token operator">=</span> i <span class="token operator">*</span> bs
        end_i <span class="token operator">=</span> start_i <span class="token operator">+</span> bs
        xb <span class="token operator">=</span> x_train<span class="token punctuation">[</span>start_i<span class="token punctuation">:</span>end_i<span class="token punctuation">]</span>
        yb <span class="token operator">=</span> y_train<span class="token punctuation">[</span>start_i<span class="token punctuation">:</span>end_i<span class="token punctuation">]</span>
        pred <span class="token operator">=</span> model<span class="token punctuation">(</span>xb<span class="token punctuation">)</span>
        loss <span class="token operator">=</span> loss_func<span class="token punctuation">(</span>pred<span class="token punctuation">,</span> yb<span class="token punctuation">)</span>

        loss<span class="token punctuation">.</span>backward<span class="token punctuation">(</span><span class="token punctuation">)</span>
        opt<span class="token punctuation">.</span>step<span class="token punctuation">(</span><span class="token punctuation">)</span>
        opt<span class="token punctuation">.</span>zero_grad<span class="token punctuation">(</span><span class="token punctuation">)</span>

<span class="token keyword">print</span><span class="token punctuation">(</span>loss_func<span class="token punctuation">(</span>model<span class="token punctuation">(</span>xb<span class="token punctuation">)</span><span class="token punctuation">,</span> yb<span class="token punctuation">)</span><span class="token punctuation">)</span>

</code></pre>
<h3 id="dataset">Dataset</h3>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">from</span> torch<span class="token punctuation">.</span>utils<span class="token punctuation">.</span>data <span class="token keyword">import</span> TensorDataset
train_ds <span class="token operator">=</span> TensorDataset<span class="token punctuation">(</span>x_train<span class="token punctuation">,</span> y_train<span class="token punctuation">)</span>
model<span class="token punctuation">,</span> opt <span class="token operator">=</span> get_model<span class="token punctuation">(</span><span class="token punctuation">)</span>

<span class="token keyword">for</span> epoch <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span>epochs<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token keyword">for</span> i <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span><span class="token punctuation">(</span>n <span class="token operator">-</span> <span class="token number">1</span><span class="token punctuation">)</span> <span class="token operator">//</span> bs <span class="token operator">+</span> <span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
        xb<span class="token punctuation">,</span> yb <span class="token operator">=</span> train_ds<span class="token punctuation">[</span>i <span class="token operator">*</span> bs<span class="token punctuation">:</span> i <span class="token operator">*</span> bs <span class="token operator">+</span> bs<span class="token punctuation">]</span>
        pred <span class="token operator">=</span> model<span class="token punctuation">(</span>xb<span class="token punctuation">)</span>
        loss <span class="token operator">=</span> loss_func<span class="token punctuation">(</span>pred<span class="token punctuation">,</span> yb<span class="token punctuation">)</span>

        loss<span class="token punctuation">.</span>backward<span class="token punctuation">(</span><span class="token punctuation">)</span>
        opt<span class="token punctuation">.</span>step<span class="token punctuation">(</span><span class="token punctuation">)</span>
        opt<span class="token punctuation">.</span>zero_grad<span class="token punctuation">(</span><span class="token punctuation">)</span>

<span class="token keyword">print</span><span class="token punctuation">(</span>loss_func<span class="token punctuation">(</span>model<span class="token punctuation">(</span>xb<span class="token punctuation">)</span><span class="token punctuation">,</span> yb<span class="token punctuation">)</span><span class="token punctuation">)</span>
</code></pre>
<p>Dataset是一个重写了__len__与__getitem__的类。<br>
一般配合下面的DataLoader一起使用</p>
<h3 id="dataloader">DataLoader</h3>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">from</span> torch<span class="token punctuation">.</span>utils<span class="token punctuation">.</span>data <span class="token keyword">import</span> DataLoader

train_ds <span class="token operator">=</span> TensorDataset<span class="token punctuation">(</span>x_train<span class="token punctuation">,</span> y_train<span class="token punctuation">)</span>
train_dl <span class="token operator">=</span> DataLoader<span class="token punctuation">(</span>train_ds<span class="token punctuation">,</span> batch_size<span class="token operator">=</span>bs<span class="token punctuation">)</span>
model<span class="token punctuation">,</span> opt <span class="token operator">=</span> get_model<span class="token punctuation">(</span><span class="token punctuation">)</span>

<span class="token keyword">for</span> epoch <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span>epochs<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token keyword">for</span> xb<span class="token punctuation">,</span> yb <span class="token keyword">in</span> train_dl<span class="token punctuation">:</span>
        pred <span class="token operator">=</span> model<span class="token punctuation">(</span>xb<span class="token punctuation">)</span>
        loss <span class="token operator">=</span> loss_func<span class="token punctuation">(</span>pred<span class="token punctuation">,</span> yb<span class="token punctuation">)</span>

        loss<span class="token punctuation">.</span>backward<span class="token punctuation">(</span><span class="token punctuation">)</span>
        opt<span class="token punctuation">.</span>step<span class="token punctuation">(</span><span class="token punctuation">)</span>
        opt<span class="token punctuation">.</span>zero_grad<span class="token punctuation">(</span><span class="token punctuation">)</span>

<span class="token keyword">print</span><span class="token punctuation">(</span>loss_func<span class="token punctuation">(</span>model<span class="token punctuation">(</span>xb<span class="token punctuation">)</span><span class="token punctuation">,</span> yb<span class="token punctuation">)</span><span class="token punctuation">)</span>
</code></pre>
<p>在上面的循环中，我们通过DataLoader的封装直接可以拿到一个batch的数据。DataLoader还具有sampler等多种功能，感兴趣可以查阅官方文档.</p>
<h4 id="做validation">做validation</h4>
<pre class=" language-python"><code class="prism  language-python">valid_ds <span class="token operator">=</span> TensorDataset<span class="token punctuation">(</span>x_valid<span class="token punctuation">,</span> y_valid<span class="token punctuation">)</span>
valid_dl <span class="token operator">=</span> DataLoader<span class="token punctuation">(</span>valid_ds<span class="token punctuation">,</span> batch_size<span class="token operator">=</span>bs <span class="token operator">*</span> <span class="token number">2</span><span class="token punctuation">)</span>

model<span class="token punctuation">,</span> opt <span class="token operator">=</span> get_model<span class="token punctuation">(</span><span class="token punctuation">)</span>

<span class="token keyword">for</span> epoch <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span>epochs<span class="token punctuation">)</span><span class="token punctuation">:</span>
    model<span class="token punctuation">.</span>train<span class="token punctuation">(</span><span class="token punctuation">)</span>
    <span class="token keyword">for</span> xb<span class="token punctuation">,</span> yb <span class="token keyword">in</span> train_dl<span class="token punctuation">:</span>
        pred <span class="token operator">=</span> model<span class="token punctuation">(</span>xb<span class="token punctuation">)</span>
        loss <span class="token operator">=</span> loss_func<span class="token punctuation">(</span>pred<span class="token punctuation">,</span> yb<span class="token punctuation">)</span>

        loss<span class="token punctuation">.</span>backward<span class="token punctuation">(</span><span class="token punctuation">)</span>
        opt<span class="token punctuation">.</span>step<span class="token punctuation">(</span><span class="token punctuation">)</span>
        opt<span class="token punctuation">.</span>zero_grad<span class="token punctuation">(</span><span class="token punctuation">)</span>

    model<span class="token punctuation">.</span><span class="token builtin">eval</span><span class="token punctuation">(</span><span class="token punctuation">)</span>
    <span class="token keyword">with</span> torch<span class="token punctuation">.</span>no_grad<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
        valid_loss <span class="token operator">=</span> <span class="token builtin">sum</span><span class="token punctuation">(</span>loss_func<span class="token punctuation">(</span>model<span class="token punctuation">(</span>xb<span class="token punctuation">)</span><span class="token punctuation">,</span> yb<span class="token punctuation">)</span> <span class="token keyword">for</span> xb<span class="token punctuation">,</span> yb <span class="token keyword">in</span> valid_dl<span class="token punctuation">)</span>

    <span class="token keyword">print</span><span class="token punctuation">(</span>epoch<span class="token punctuation">,</span> valid_loss <span class="token operator">/</span> <span class="token builtin">len</span><span class="token punctuation">(</span>valid_dl<span class="token punctuation">)</span><span class="token punctuation">)</span>
</code></pre>
<p>注意这里的model.train(), model.eval()，一定不能遗漏！BatchNorm与Dropout层在train-time, test-time有不同的行为，所以要根据这些flag进行判断。</p>
<h2 id="sequential-api">Sequential API</h2>
<p>非常类似Keras,最高阶的API</p>
<pre class=" language-python"><code class="prism  language-python">model <span class="token operator">=</span> nn<span class="token punctuation">.</span>Sequential<span class="token punctuation">(</span>
    nn<span class="token punctuation">.</span>Conv2d<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">16</span><span class="token punctuation">,</span> kernel_size<span class="token operator">=</span><span class="token number">3</span><span class="token punctuation">,</span> stride<span class="token operator">=</span><span class="token number">2</span><span class="token punctuation">,</span> padding<span class="token operator">=</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
    nn<span class="token punctuation">.</span>ReLU<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
    nn<span class="token punctuation">.</span>Conv2d<span class="token punctuation">(</span><span class="token number">16</span><span class="token punctuation">,</span> <span class="token number">16</span><span class="token punctuation">,</span> kernel_size<span class="token operator">=</span><span class="token number">3</span><span class="token punctuation">,</span> stride<span class="token operator">=</span><span class="token number">2</span><span class="token punctuation">,</span> padding<span class="token operator">=</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
    nn<span class="token punctuation">.</span>ReLU<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
    nn<span class="token punctuation">.</span>Conv2d<span class="token punctuation">(</span><span class="token number">16</span><span class="token punctuation">,</span> <span class="token number">10</span><span class="token punctuation">,</span> kernel_size<span class="token operator">=</span><span class="token number">3</span><span class="token punctuation">,</span> stride<span class="token operator">=</span><span class="token number">2</span><span class="token punctuation">,</span> padding<span class="token operator">=</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
    nn<span class="token punctuation">.</span>ReLU<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
    nn<span class="token punctuation">.</span>AdaptiveAvgPool2d<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
    Lambda<span class="token punctuation">(</span><span class="token keyword">lambda</span> x<span class="token punctuation">:</span> x<span class="token punctuation">.</span>view<span class="token punctuation">(</span>x<span class="token punctuation">.</span>size<span class="token punctuation">(</span><span class="token number">0</span><span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span>
</code></pre>

