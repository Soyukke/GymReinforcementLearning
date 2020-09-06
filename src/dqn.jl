using Flux
import Flux.@functor
export DQNetwork, loss, run_episode
"""
DQNネットワーク
"""
struct DQNetwork
    nstatetype::Integer
    naction::Integer
    model 
end
@functor DQNetwork

"""
Deep Q-Networkインスタンス作成
DQNモデル
s -> Q(s, a1), Q(s, a2), Q(s, a3)
"""
function DQNetwork()
    # 状態ベクトルの次元数
    nstate_type = 2
    # 行動の種類
    naction = 3
    model = Chain(
        Dense(nstate_type, 32, relu),
        Dense(32, 128, relu),
        Dense(128, naction)
    )
    return DQNetwork(nstate_type, naction, model)
end

"""
loss function
損失関数
T(s, a) -> s'
l = Q(s, a) -  R(s) + max_p Q(s', p)
L = (1/2) * l^2
"""
function loss(q::DQNetwork)
    batch_size = 10
    state = rand(q.nstatetype, batch_size)
    # (naction, batch_size)
    qa = q.model(state)
    # 最大値 max Q(s, a)を取得する
end

function qs(q::DQNetwork, s::Vector{Float32})
    predicts = q.model(s)
    return predicts
end

"""
行動aを選択する
"""
function action(q::DQNetwork, s::Vector{Float32})::Integer
    # 行動a
    a = argmax(qs(q, s))
    return a
end

"""
行動aを選択する
"""
function actions(q::DQNetwork)::Vector{Integer}
    batch_size = 10
    state = rand(q.nstatetype, batch_size)
    # (naction, batch_size)
    qs = q.model(state)
    # 行動a
    as = argmaxs(qs)
    return as
end

"""
行列mの縦方向ごとに見て，argmaxを計算する
# Examples
m
```
3×10 Array{Float32,2}:
 -0.0407064  -0.0679783  -0.0769743  -0.0316168   -0.0605061  -0.00688636  -0.07329     -0.0785109  -0.0417965  -0.0376177
 -0.0545554  -0.0453474  -0.0330656   0.00389661  -0.0515486  -0.00836701  -0.00768139  -0.0179061  -0.0526757  -0.05496
  0.189985    0.281258    0.261014    0.0419651    0.275313    0.0175994    0.144541     0.197473    0.211431    0.162823
````
出力
```julia
1×10 Array{Int64,2}:
 3  3  3  3  3  3  3  3  3  3
end
"""
function argmaxs(m::Matrix)::Vector{Integer}
    return mapslices(argmax, m, dims=1)
end

"""
1エピソード実行する
ペアでデータを保持しておく: s_i, a_i, R(s_i), max_p Q(s_{i+1}, p)
"""
function run_episode(dqn::DQNetwork)
    gym = Gym()
    env = initenv(gym)
    s0 = Float32.(reset(env))
    data = []
    while true
        qs0 = qs(dqn, s0)
        a0 = argmax(qs0)
        r0 = 0
        maxq0 = qs0[a0]
        s1, r1, done, info = step(env, a0)
        s1 = Float32.(s1)
        if done
            break
        end
        qs1 = qs(dqn, s1)
        a1 = argmax(qs1)
        maxq1 = qs1[a1]
        # 学習に必要なパラメータ
        record = [s0, a0, r0, maxq1]
        push!(data, record)
        # 更新する
        s0 = s1
        r0 = r1
    end
    return data
end

"""
DQNの実行流れ
# Simulation step
1. DQNでsからa ∈ A, Q(s, p)が最大なaをQから予測する
2. aを実行し，T(s, a) -> s'を得る
3. s = s'として1へ戻る．episodeが終了したら終わる

1~3で1エピソードのシミュレーションが終了する

# Prepare step
学習データを作成する
ペアでデータを保持しておく: s_i, a_i, R(s_i), max_p Q(s_{i+1}, p)

# Training step
以下の損失関数を使ってパラメータを学習する．backwardの対象となるのはQ(s_i, a_i)で，max_p Q(s_{i+1}, p)は定数として扱う
model(s_i)の出力をa_iでmaskし，以下の損失関数を計算する
L = (1/2) * (R(s_i) + γ * max_p Q(s_{i+1}, p) - Q(s_i, a_i))^2
"""
function dqnmain()
end