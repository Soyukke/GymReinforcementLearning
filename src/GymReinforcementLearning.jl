module GymReinforcementLearning

export Gym, initenv, example

using PyCall
using Conda

struct Gym
    gym::PyObject
    envs::PyObject
end

struct Environment
    env::PyObject
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
行動を決定する
"""
function decide_action()
end

"""
報酬を評価する
"""
function calculate_reward()
end

"""
行動する
状態が変化する
"""
function execute_action()
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

reset(env::Environment) = env.env.reset()
sample(env::Environment) = env.env.action_space.sample()
close(env::Environment) = env.env.close()
step(env::Environment, action) = env.env.step(action)
render(env::Environment) = env.env.render()

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

end # module
