Adjoint Method를 활용한 Optimal Control Problem 풀이
=================================================================

Solving Optimal Control Problem
-------------------------------

다음과 같은 Contrained Optimal Control Problem을 Lagrage Multiplier를 통해 푸는 방법을 소개합니다 [#Lenhart2007]_.

.. math::

   \min_u \int_{t_0}^{t_1} f(t, x(t), u(t))~dt

subject to

.. math::

   \begin{cases}
   x'(t) &= g(t,x(t), u(t))\\
   x(t_0)&=x_0
   \end{cases}

가장 간단한 방법론은 다음과 같습니다.

#. 해당 Optimal Control Problem의 Hamiltonian을 구성한다.

   .. math::
      H = f(t, x(t), u(t)) + \lambda(t)\cdot g(t, x(t), u(t))

#. Optimality condition

   .. math::
      \frac{\partial H}{\partial u} = 0 \text{ at } u^* \Rightarrow f_u + \lambda g_u = 0

#. Adjoint equation

   .. math::
      \lambda' = -\frac{\partial H}{\partial x} \Rightarrow \lambda' = -(f_x + \lambda g_x)

#. Transversality condition

   .. math::
      \lambda(t_1) = 0

#. State equation

   .. math::
      x' = g(t,x,u) = \frac{\partial H}{\partial \lambda}, ~x(t_0)=x_0


참고로 Pontryagin's maximum principle은 다음과 같습니다.

.. note::

   If :math:`u^*(t)` and :math:`x^*(t)` are optimal for the above optimal control problem, then there exists a piecewise differentiable adjoint variable :math:`\lambda(t)` such that

   .. math::

      H(t, x^*(t), u(t), \lambda(t)) \leq H(t, x^*(t), u^*(t), \lambda(t))

   for all controls :math:`u` at each time :math:`t`, where the *Hamiltonian* :math:`H` is

   .. math::
      H = f(t, x(t), u(t)) + \lambda(t)\cdot g(t, x(t), u(t))

   and

   .. math::
      \lambda'(t) & = - \frac{\partial H(t, x^*(t), u^*(t), \lambda(t))}{\partial x}\\
      \lambda(t_1) &= 0

Example 1
---------

.. math::
   \min_u \int_0^1 u^2(t)dt

subject to

.. math::
   x'(t) &= x(t) + u(t)\\
   x(0) &= 1

.. note::

   #. 해당 Optimal Control Problem의 Hamiltonian을 구성한다.

      .. math::
         H &= f(t, x(t), u(t)) + \lambda(t)\cdot g(t, x(t), u(t))\\
           &= u^2 + \lambda(x+u)

   #. Optimality condition

      .. math::
         \frac{\partial H}{\partial u} &= 0 \text{ at } u^* \Rightarrow f_u + \lambda g_u = 0\\
         2u + \lambda &=0 \Rightarrow u^* = -\frac{1}{2} \lambda

   #. Adjoint equation

      .. math::
         \lambda' &= -\frac{\partial H}{\partial x} \Rightarrow \lambda' = -(f_x + \lambda g_x)\\
         &=-\lambda \Rightarrow \lambda(t) = c \cdot e^{-t}

   #. Transversality condition

      .. math::
         \lambda(t_1) &= 0\\
         \lambda(1) &= c\cdot e^{-1} = 0 \\ &\Rightarrow c=0 \\ &\Rightarrow \lambda=0 \\ &\Rightarrow u^*=-\frac{1}{2}\lambda = 0

   #. State equation

      .. math::
         {x^{*}}' &= g(t,x^*,u) = \frac{\partial H}{\partial \lambda}, ~x^*(t_0)=x_0\\
         {x^{*}}' &= g(t,x^*,u) = \frac{\partial H}{\partial \lambda}, ~x^*(0)=1\\
         x^* &= e^t

Example 2: Forward-Backward Sweep Method
----------------------------------------

첫번째 예제와 같이 :math:`H_u=0 \text{ at } u^*` 를 통해서 3개의 Unknowns를 2개로 소거하는 경우는 많지 않습니다.
그런 경우에는 Forward-Backward Sweep [#Lenhart2007]_ 이라는 Iterative Scheme을 사용합니다.

#. :math:`u^*` 의 초기추정값(:math:`u_0`)을 설정합니다.
#. 초기추정값(:math:`u_0`)을 이용하여 :math:`x_0` 의 방정식인 State differential equation을 풉니다. (Forward)
#. 앞에서 구한 :math:`u_0` 와 :math:`u_0` 를 활용하여 :math:`\lambda_0` 의 방정식인 Adjoint differential equation을 풉니다. (Backward)
#. :math:`H_u(u_{new}) = 0` 을 풀어서 얻은 :math:`u_{new}` 과 :math:`u_0` 을 *적절한* 방식으로 :math:`u_1` 을 업데이트 합니다.

   .. math::
      u_1 = \frac{1}{2}\left(u_{new} + u_0\right)

#. Convergence를 체크하고, 수렴할때까지 위의 단계들을 반복합니다.

.. [#Lenhart2007] Lenhart, Suzanne, and John T. Workman. Optimal control applied to biological models. Chapman and Hall/CRC, 2007.