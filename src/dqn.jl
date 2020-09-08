using Flux
using Flux.Data
using Random
import Flux.@functor, Flux.onehotbatch
import Statistics.mean, Statistics.std
import Base.Iterators.partition
export DQNetwork, loss, run_episode, dqnmain, huber_loss

include("replay.jl")

"""
DQNネットワーク
"""
struct DQNetwork
    nstatetype::Integer
    naction::Integer
    env::Environment
    model 
end
@functor DQNetwork

"""
Deep Q-Networkインスタンス作成
DQNモデル
s -> Q(s, a1), Q(s, a2), Q(s, a3)
## 状態ベクトルの次元数
nstate_type
## 行動の種類
nation
"""
function DQNetwork(env::Environment)
    obs = observation_space(env)
    action = action_space(env)
    nstate_type = n_state_features(obs)
    naction = n_action(action)
    @show nstate_type, naction

    model = Chain(
        Dense(nstate_type, 32, relu),
        Dense(32, 128, relu),
        Dense(128, naction)
    )
    return DQNetwork(nstate_type, naction, env, model)
end

"""
loss function
損失関数
T(s, a) -> s'
l = Q(s, a) -  R(s) + max_p Q(s', p)
L = (1/2) * l^2
"""
# function loss(q::DQNetwork)
#     batch_size = 10
#     state = rand(q.nstatetype, batch_size)
#     # (naction, batch_size)
#     qa = q.model(state)
#     # 最大値 max Q(s, a)を取得する
# end

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
function run_episode(dqn::DQNetwork; is_render=false)
    env = dqn.env
    s0 = Float32.(reset(env))
    data = []
    r0 = 0
    while true
        if is_render
            render(env)
        end
        qs0 = qs(dqn, s0)
        a0 = argmax(qs0)
        # ε-greedy
        a_actual = rand() < 0.1 ? rand(range(1, stop=dqn.naction)) : a0
        maxq0 = qs0[a0]
        s1, r1, done, info = step(env, a_actual-1)
        s1 = Float32.(s1)
        if done
            break
        end
        qs1 = qs(dqn, s1)
        a1 = argmax(qs1)
        maxq1 = qs1[a1]
        # 学習に必要なパラメータ
        record = [s0, a_actual, r0, maxq1]
        push!(data, record)
        # 更新する
        s0 = s1
        r0 = r1
    end
    return data
end

function huber_loss(x::Number)
    x = Float32(x)
    δ::Float32 = 1
    if abs(x) ≤ δ
        return 0.5f0 * x^2
    else
        return δ * (abs(x) - 0.5f0 * x)
    end
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
    γ::Float32 = 0.9
    gym = Gym()
    env = initenv(gym)
    dqn = DQNetwork(env)
    ps = params(dqn)
    opt = ADAM(1e-3)
    minibatchsize = 500
    function loss(x, a, y)
        return L = mean((1f0/2.0f0) * huber_loss.(dqn.model(x).*a - y))
    end
    # function loss(x)
    #     @show size(x)
    #     # return L = mean((1f0/2.0f0) * huber_loss.(dqn.model(x).*a - y))
    # end

    try
        isrender = false
        for i in 1:1000
            # data = run_episode(dqn, is_render=false)
            # ReplayBufferにためる
            # buffer
            replaydata = ReplayBuffer()
            isendreplay = true
            # 平均継続数
            meantime = 0
            numdata = 0
            while isendreplay
                data = run_episode(dqn, is_render=isrender)
                meantime += length(data)
                numdata += 1
                for record in data
                    isendreplay = add_record(replaydata, record)
                    !isendreplay && break
                end
            end
            meantime /= numdata

            data = replaydata.data
            @assert length(data) == replaydata.maxsize
            # 入力と学習データを作詞絵する
            state_list = map(record->record[begin], data)
            states = reduce(hcat, state_list)
            # μ = mean(states)
            # σ = std(states)
            # states = (states .- μ) / σ

            action_list = map(record->record[begin+1], data)
            # onehot行列でmask用行列
            actions = onehotbatch(action_list, range(1, stop=dqn.naction))
            # max_p Q(s, p)の予測値
            q_s1_a1 = dqn.model(states) .* actions
            # (R(s) + max_p Q(s_{i+1}, p)) - Q(s, a)
            q_s2_p = Float32.(reduce(hcat, map(record->record[end], data)))
            println(q_s2_p[begin])
            r_s1 = Float32.(reduce(hcat, map(record->record[end-1], data)))
            # 教師データ
            y_matrix = repeat(r_s1 + γ*q_s2_p, dqn.naction)
            # normalize
            # μ = mean(y_matrix)
            # σ = std(y_matrix)
            # y_matrix = (y_matrix .- μ) / σ
            indices_shuffled = randperm(replaydata.maxsize)
            @show size(states), size(actions), size(y_matrix)
            traindata = (states, actions, y_matrix)
            traindata = DataLoader(traindata, batchsize=minibatchsize, shuffle=true)

            # 学習データはこんな感じで用意する
            # traindata = zip((states, ), (actions, ), (y_matrix, ))
            # L = mean((1f0/2.0f0) * (q_s1_a1 - repeat(r_s1 + γ*q_s2_p, naction)).^2)
            Flux.train!(loss, ps, traindata, opt)
            l = loss(states, actions, y_matrix)
            @info i size(states, 2), l, meantime
        end
    catch e
        print(e)
        close(env)
    end
end