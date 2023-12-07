"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
"""
# from . import cmudict

_pad        = "_"
_eos        = "~"
_characters = "ا ب پ ت ٹ ث ج چ ح خ د ڈ ذ ر ڑ ز ژ س ش ص ض ط ظ ع غ ف ق ک گ ل م ن ں ه و ء آ ئ ؤ ء ى ي ے َ ً ُ ٌ ِ ٍ ّ ۔ ۰ ۱ ۲ ۳ ۴ ۵ ۶ ۷ ۸ ۹ ! ؟ ‘ ’ “ ” ، ۔ - ـ : ؛"

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
#_arpabet = ["@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad, _eos] + list(_characters) #+ _arpabet
