import pytest
import torch

from maic.returns import discounted_returns


def test_terminal_reward_included_without_terminal_bootstrap():
    result = discounted_returns(torch.tensor([[2., 3.]]), torch.tensor([[False, True]]), torch.tensor([999.]), .9)
    torch.testing.assert_close(result, torch.tensor([[4.7, 3.]]))


def test_truncation_retains_correct_endpoint_bootstrap():
    result = discounted_returns(torch.tensor([[2., 3.]]), torch.zeros(1, 2, dtype=torch.bool), torch.tensor([10.]), .9)
    torch.testing.assert_close(result, torch.tensor([[12.8, 12.]]))


def test_padding_neither_discounts_nor_consumes_nan_rewards():
    result = discounted_returns(torch.tensor([[2., 3., float("nan")]]), torch.zeros(1, 3, dtype=torch.bool),
                                torch.tensor([10.]), .9, torch.tensor([[True, True, False]]))
    torch.testing.assert_close(result, torch.tensor([[12.8, 12., 0.]]))


@pytest.mark.parametrize("gamma", [-.1, 1., float("nan")])
def test_invalid_discount_rejected(gamma):
    with pytest.raises(ValueError):
        discounted_returns(torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.bool), torch.zeros(1), gamma)


def test_internal_padding_rejected():
    with pytest.raises(ValueError, match="right padding"):
        discounted_returns(torch.ones(1, 3), torch.zeros(1, 3, dtype=torch.bool), torch.zeros(1), .9,
                           torch.tensor([[True, False, True]]))
