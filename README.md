# JuliaでGym

## 環境

- Julia 1.3.1

gymインストール．
```julia
using PyCall
using Conda
Conda.add("gym", channel="conda-forge")
```

## gymを使う

Pythonから使うのと同じ感じでそのまま使える

```julia
gym = pyimport("gym")
env = gym.make("MountainCar-v0")
# 初期化
observation = env.reset()
# 描画
env.render()
```

## 参考
https://qiita.com/ishizakiiii/items/75bc2176a1e0b65bdd16#openai-gym-%E4%BD%BF%E3%81%84%E6%96%B9