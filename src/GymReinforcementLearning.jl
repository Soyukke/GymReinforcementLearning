module GymReinforcementLearning
export Gym, initenv, example, QLearning
export reset, sample, close, step, render, observation_space
export low, high, n_state_features
export decide_action, execute_action
export main

using PyCall
using Conda

"""
pyimportしたgymのインスタンス
"""
struct Gym
    gym::PyObject
    envs::PyObject
end

"""
シミュレーション環境インスタンス
"""
struct Environment
    env::PyObject
end

struct Observation
    obs::PyObject
end

struct Action
    action::PyObject
end

"""
Q学習構造体
"""
struct QLearning
    env::Environment
    qmat::Array
end

"""
環境情報から行列を初期化する
行列サイズ (離散状態1数, 離散状態2数, ..., 行動数)
"""
function QLearning(env; nstep=20)
    obs = observation_space(env)
    act = action_space(env)
    naction = n_action(act)
    # 状態数はstep^状態ベクトルサイズ
    nfeatures = n_state_features(obs)
    qmat = zeros(repeat([nstep], nfeatures)..., naction)
    return QLearning(env, qmat)
end

function indices(q::QLearning, state, a)
    naction = action_space(q.env) |> n_action
    # 現在の状態sから方策πにより行動aを決定する
    nstep = size(q.qmat)[1]
    obs = observation_space(q.env)
    # 状態の最小値と最大値を使ってnormalizeする
    state_low = low(obs)
    state_high = high(obs)
    nfeatures = n_state_features(obs)
    @assert size(state_low) == size(state_high) == size(state)
    # @info "low, high", state_low, state_high
    state_normal = @. (state - state_low) / (state_high - state_low)
    # 何行目の状態に対応するのか
    # @info "step", 1 / nstep
    # 刻み幅
    step = 1/nstep
    idx_state = ceil.(Int, state_normal / step)
    return [idx_state..., a+1]
end

"""
状態s, 行動aの行動価値を取得する
"""
function (q::QLearning)(state, a)
    idx = indices(q, state, a)
    # s, a
    return q.qmat[idx...]
end

"""
Actionを決定する
"""
function decide_action(q::QLearning, state::Vector)
    naction = action_space(q.env) |> n_action
    # sに対応する行を取得
    qs = [q(state, 0), q(state, 1), q(state, 2)]
    # ϵ-greedy法
    # ϵ = 0.3
    # if rand() < ϵ
    #     @info "random action"
    #     return rand(0:naction-1)
    # else
    #     return argmax(qs) - 1
    # end

    # greedy
    return argmax(qs) - 1

    # softmax法
    # return argmax(qs / sum(exp.(qs)))
end

"""
行動する
状態が変化する

return s2, r2, done
"""
function execute_action(q::QLearning, s1, r1, action)
    # 変化前の状態sで行動aをとったときのQ(s, a)を更新する
    s2, r2, done, info = step(q.env, action)
    # 変化後の状態s'からmax_p Q(s', p)を取得して，Q(s, a)を更新する
    # @info s2, reward, done, info
    # Q-Matrixを更新するs
    Q1 = q(s1, action)
    maxQ2 = maximum([q(s2, 0), q(s2, 1), q(s2, 2)])
    γ = 0.9
    α = 0.4
    # qmat更新
    q.qmat[indices(q, s1, action)...] += α * (r1 + γ*maxQ2 - Q1)
    return s2, r2, done
end

function Gym()
    try
        pyimport("gym")
    catch
        println("gymをインストール")
        Conda.add("gym", channel="conda-forge")
    end
    gym = pyimport("gym")
    envs = pyimport("gym.envs")
    return Gym(gym, envs)
end

"""
環境を初期化する
[座標, 速度]
"""
function initenv(s::Gym)
    env = s.gym.make("MountainCar-v0")
    return Environment(env)
end

"""
報酬を評価する
"""
function calculate_reward()
end

"""
状態sで行動aを取った後の状態s'が不明な問題の場合
行動価値関数Q(s, a)
状態sで行動aをとったときに得られる将来を加味した報酬
R(s')はアルゴリズムで計算できるとする
R(s') + γ max[Q(s', a')]
## 入力
状態sにおける報酬r, 状態s, 行動a, 遷移後状態s'

## 出力

"""
function Q()
end

"""
シミュレーション環境関数
"""
Base.reset(env::Environment) = env.env.reset()
sample(env::Environment) = env.env.action_space.sample()
Base.close(env::Environment) = env.env.close()
step(env::Environment, action) = env.env.step(action)
render(env::Environment) = env.env.render()
observation_space(env::Environment) = Observation(env.env.observation_space)
action_space(env::Environment) = Action(env.env.action_space)

"""
状態変数の定義域
"""
low(obs::Observation)::Vector = obs.obs.low
high(obs::Observation)::Vector = obs.obs.high
n_state_features(obs::Observation)::Integer = first(obs.obs.shape)

"""
Actionの種類
"""
n_action(a::Action) = a.action.n

"""
サンプルしたアクションを実行し，描画し続ける
"""
function example()
    gym = Gym()
    env = initenv(gym)
    reset(env)
    while true
        render(env)
        action = sample(env)
        observation, reward, done, info = step(env, action)
        if done
            break
        end
    end
    close(env)
end

function main()
    gym = Gym()
    env = initenv(gym)
    q = QLearning(env)
    try
        for i in 1:1000
            @info i
            s0 = reset(q.env)
            s = copy(s0)
            r = 0
            while true
                render(q.env)
                a = decide_action(q, s)
                s, r, done = execute_action(q, s, r, a)
                if done
                    break
                end
            end
        end
    catch e
        print(e)
    end
    close(env)
end

end # module
