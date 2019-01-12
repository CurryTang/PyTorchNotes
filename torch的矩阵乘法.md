---


---

<h1 id="torch的矩阵乘法">torch的矩阵乘法</h1>
<p>torch中的矩阵乘法有两个api:</p>
<ul>
<li><a href="http://torch.mm">torch.mm</a></li>
<li>torch.matmul</li>
</ul>
<p>其中，torch.mm是没有考量broadcasting的，所以一般情况下我们只需要考虑使用torch.matmul即可。在一维的情况下，torch.matmul就是返回向量之间的dot product;对于两个普通的矩阵，其返回的就是普通的矩阵乘法结果；而当到了高维的情况，torch.matmul会进行batch之间的矩阵乘法。高维的情况在实际中非常常见，当我们需要在两个batch之间做矩阵乘法时，torch.matmul会自动帮我们完成broadcasting的任务。</p>
<pre class=" language-python"><code class="prism  language-python">In <span class="token punctuation">[</span><span class="token number">51</span><span class="token punctuation">]</span><span class="token punctuation">:</span> mat1 <span class="token operator">=</span> torch<span class="token punctuation">.</span>randn<span class="token punctuation">(</span><span class="token number">3</span><span class="token punctuation">)</span>

In <span class="token punctuation">[</span><span class="token number">52</span><span class="token punctuation">]</span><span class="token punctuation">:</span> torch<span class="token punctuation">.</span>matmul<span class="token punctuation">(</span>mat1<span class="token punctuation">,</span> mat1<span class="token punctuation">)</span>
Out<span class="token punctuation">[</span><span class="token number">52</span><span class="token punctuation">]</span><span class="token punctuation">:</span> tensor<span class="token punctuation">(</span><span class="token number">2.5802</span><span class="token punctuation">)</span>

In <span class="token punctuation">[</span><span class="token number">53</span><span class="token punctuation">]</span><span class="token punctuation">:</span> mat2 <span class="token operator">=</span> torch<span class="token punctuation">.</span>randn<span class="token punctuation">(</span><span class="token number">3</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">)</span>

In <span class="token punctuation">[</span><span class="token number">54</span><span class="token punctuation">]</span><span class="token punctuation">:</span> torch<span class="token punctuation">.</span>matmul<span class="token punctuation">(</span>mat2<span class="token punctuation">,</span> mat2<span class="token punctuation">)</span>
Out<span class="token punctuation">[</span><span class="token number">54</span><span class="token punctuation">]</span><span class="token punctuation">:</span> 
tensor<span class="token punctuation">(</span><span class="token punctuation">[</span><span class="token punctuation">[</span> <span class="token number">3.1535</span><span class="token punctuation">,</span>  <span class="token number">1.9675</span><span class="token punctuation">,</span> <span class="token operator">-</span><span class="token number">0.3797</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
        <span class="token punctuation">[</span> <span class="token number">1.3713</span><span class="token punctuation">,</span>  <span class="token number">1.2634</span><span class="token punctuation">,</span> <span class="token operator">-</span><span class="token number">0.0071</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
        <span class="token punctuation">[</span><span class="token operator">-</span><span class="token number">2.3827</span><span class="token punctuation">,</span> <span class="token operator">-</span><span class="token number">1.5281</span><span class="token punctuation">,</span>  <span class="token number">0.9703</span><span class="token punctuation">]</span><span class="token punctuation">]</span><span class="token punctuation">)</span>

In <span class="token punctuation">[</span><span class="token number">55</span><span class="token punctuation">]</span><span class="token punctuation">:</span> mat3 <span class="token operator">=</span> torch<span class="token punctuation">.</span>randn<span class="token punctuation">(</span><span class="token number">10</span><span class="token punctuation">,</span><span class="token number">3</span><span class="token punctuation">,</span><span class="token number">3</span><span class="token punctuation">)</span>

In <span class="token punctuation">[</span><span class="token number">56</span><span class="token punctuation">]</span><span class="token punctuation">:</span> torch<span class="token punctuation">.</span>matmul<span class="token punctuation">(</span>mat3<span class="token punctuation">,</span> mat3<span class="token punctuation">)</span><span class="token punctuation">.</span>size<span class="token punctuation">(</span><span class="token punctuation">)</span>
Out<span class="token punctuation">[</span><span class="token number">56</span><span class="token punctuation">]</span><span class="token punctuation">:</span> torch<span class="token punctuation">.</span>Size<span class="token punctuation">(</span><span class="token punctuation">[</span><span class="token number">10</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">]</span><span class="token punctuation">)</span> <span class="token comment"># 注意这里第一维是保留不变的</span>
</code></pre>

