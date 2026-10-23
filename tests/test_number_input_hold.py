"""DOM event regressions. Install quickjs to run without a browser."""
from pathlib import Path
import unittest

try:
    import quickjs
except ImportError:
    quickjs = None


@unittest.skipIf(quickjs is None, "Optional quickjs package is not installed")
class NumberInputHoldTests(unittest.TestCase):
    def setUp(self):
        self.js = quickjs.Context()
        self.js.eval('''
            const listeners = {};
            let timers = new Map(), nextTimer = 0, value = 30;
            const window = {addEventListener: (type, fn) => listeners[type] = fn};
            const document = {
                hidden: false,
                addEventListener: (type, fn) => listeners[type] = fn,
                getElementById: () => input,
            };
            const setTimeout = fn => {timers.set(++nextTimer, fn); return nextTimer;};
            const clearTimeout = id => timers.delete(id);
            const widget = {querySelector: selector => selector === 'input' ? input : button};
            const input = {id: 'age', closest: () => widget};
            const makeButton = () => ({
                disabled: false,
                dataset: {testid: 'stNumberInputStepUp'},
                closest: selector => selector === '[data-testid="stNumberInput"]' ? widget : button,
                click: () => value++,
                getBoundingClientRect: () => ({left: 0, right: 20, top: 0, bottom: 20}),
            });
            let button = makeButton();
            const outside = {closest: () => null};
            const fire = (type, target = button, extra = {}) => {
                const event = {target, button: 0, isPrimary: true, pointerId: 1,
                    isTrusted: true, clientX: 10, clientY: 10, prevented: false,
                    preventDefault() {this.prevented = true;},
                    stopImmediatePropagation() {this.stopped = true;}, ...extra};
                listeners[type](event);
                return event;
            };
            const tick = () => {
                const pending = [...timers.values()]; timers.clear();
                pending.forEach(fn => fn());
            };
        ''')
        self.script = (Path(__file__).resolve().parents[1] / "app/static/number_input_hold.html").read_text()
        self.script = self.script.removeprefix("<script>").strip().removesuffix("</script>")
        self.js.eval(self.script)

    def test_first_hold_repeats_and_release_stops_without_extra_step(self):
        self.js.eval("fire('pointerdown'); tick(); tick();")
        self.assertEqual(self.js.eval("value"), 33)
        self.js.eval("fire('pointerup'); fire('click'); tick();")
        self.assertEqual(self.js.eval("value"), 33)
        self.assertEqual(self.js.eval("timers.size"), 0)

    def test_expander_click_is_never_blocked(self):
        self.js.eval("fire('pointerdown', outside)")
        self.assertFalse(self.js.eval("fire('click', outside).prevented"))
        self.js.eval("fire('pointerdown'); tick(); fire('pointerup');")
        self.assertFalse(self.js.eval("fire('click', outside).prevented"))

    def test_hold_survives_script_rerun_and_replaced_button(self):
        self.js.eval("fire('pointerdown'); button = makeButton();")
        self.js.eval(self.script)
        self.js.eval("tick(); tick();")
        self.assertEqual(self.js.eval("value"), 33)

    def test_disabled_button_and_cancellation(self):
        self.js.eval("button.disabled = true; fire('pointerdown'); tick();")
        self.assertEqual(self.js.eval("value"), 30)
        self.js.eval("button.disabled = false; fire('pointerdown'); fire('pointercancel'); tick();")
        self.assertEqual(self.js.eval("value"), 31)

    def test_leave_and_blur_stop_repeat(self):
        self.js.eval("fire('pointerdown'); fire('pointermove', button, {clientX: 30}); tick();")
        self.assertEqual(self.js.eval("value"), 31)
        self.js.eval("fire('pointerdown'); fire('blur'); tick();")
        self.assertEqual(self.js.eval("value"), 32)


if __name__ == '__main__':
    unittest.main()
