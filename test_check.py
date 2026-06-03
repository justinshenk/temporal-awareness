"""Quick test of the AttributionSettings fixes."""
import sys
sys.path.insert(0, '/sessions/festive-zen-mccarthy/mnt/temporal-awareness/src')

from attribution_patching import AttributionSettings

# Test 1: test_default_settings
print("Test 1: test_default_settings")
settings = AttributionSettings()
assert "standard" in settings.methods, f"Expected 'standard' in {settings.methods}"
assert "eap" in settings.methods, f"Expected 'eap' in {settings.methods}"
print("  PASS: default settings contain 'standard' and 'eap'")

# Test 2: test_standard_only
print("Test 2: test_standard_only")
settings = AttributionSettings.standard_only()
assert settings.methods == ["standard"], f"Expected ['standard'], got {settings.methods}"
print("  PASS: standard_only() returns ['standard']")

# Test 3: test_with_ig
print("Test 3: test_with_ig")
settings = AttributionSettings.with_ig(steps=20)
assert "eap_ig" in settings.methods, f"Expected 'eap_ig' in {settings.methods}"
assert settings.ig_steps == 20, f"Expected ig_steps=20, got {settings.ig_steps}"
print("  PASS: with_ig(steps=20) sets eap_ig and ig_steps correctly")

print("\nAll tests passed!")
