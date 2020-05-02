#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines autosub's command line functionality.
"""
# pylint: disable=too-many-lines
# Import built-in modules
import gettext
import os
import sys
import subprocess
import tempfile
import gc
import json

# Import third-party modules
import auditok
import googletrans
import langcodes
import pysubs2

# Any changes to the path and your own modules
from autosub import constants
from autosub import core
from autosub import exceptions
from autosub import ffmpeg_utils
from autosub import lang_code_utils
from autosub import sub_utils
from autosub import api_google
from autosub import api_baidu

CMDLINE_UTILS_TEXT = gettext.translation(domain=__name__,
                                         localedir=constants.LOCALE_PATH,
                                         languages=[constants.CURRENT_LOCALE],
                                         fallback=True)

_ = CMDLINE_UTILS_TEXT.gettext


def list_args(args):
    """
    Check if there's any list args.
    """
    if args.list_formats:
        print(_("List of output formats:\n"))
        print("{column_1}{column_2}".format(
            column_1=lang_code_utils.wjust(_("Format"), 18),
            column_2=_("Description")))
        for subtitles_format, format_description in sorted(constants.OUTPUT_FORMAT.items()):
            print("{column_1}{column_2}".format(
                column_1=lang_code_utils.wjust(subtitles_format, 18),
                column_2=format_description))
        print(_("\nList of input formats:\n"))
        print("{column_1}{column_2}".format(
            column_1=lang_code_utils.wjust(_("Format"), 18),
            column_2=_("Description")))
        for subtitles_format, format_description in sorted(constants.INPUT_FORMAT.items()):
            print("{column_1}{column_2}".format(
                column_1=lang_code_utils.wjust(subtitles_format, 18),
                column_2=format_description))
        return True

    if args.detect_sub_language:
        print(_("Use py-googletrans to detect a sub file's first line language."))
        pysubs2_obj = pysubs2.SSAFile.load(args.detect_sub_language)
        translator = googletrans.Translator(
            user_agent=args.user_agent,
            service_urls=args.service_urls)
        result_obj = translator.detect(pysubs2_obj.events[0].text)
        print("{column_1}{column_2}".format(
            column_1=lang_code_utils.wjust(_("Lang code"), 18),
            column_2=_("Confidence")))
        print("{column_1}{column_2}\n".format(
            column_1=lang_code_utils.wjust(result_obj.lang, 18),
            column_2=result_obj.confidence))
        args.list_speech_codes = result_obj.lang

    if args.list_speech_codes:
        if args.list_speech_codes == ' ':
            print(_("List of all lang codes for speech-to-text:\n"))
            print("{column_1}{column_2}".format(
                column_1=lang_code_utils.wjust(_("Lang code"), 18),
                column_2=_("Description")))
            for code, language in sorted(constants.SPEECH_TO_TEXT_LANGUAGE_CODES.items()):
                print("{column_1}{column_2}".format(
                    column_1=lang_code_utils.wjust(code, 18),
                    column_2=language))
        else:
            print(_("Match Google Speech-to-Text lang codes."))
            lang_code_utils.match_print(
                dsr_lang=args.list_speech_codes,
                match_list=list(constants.SPEECH_TO_TEXT_LANGUAGE_CODES.keys()),
                min_score=args.min_score)
        return True

    if args.list_translation_codes:
        if args.list_translation_codes == ' ':
            print(_("List of all lang codes for translation:\n"))
            print("{column_1}{column_2}".format(
                column_1=lang_code_utils.wjust(_("Lang code"), 18),
                column_2=_("Description")))
            for code, language in sorted(constants.TRANSLATION_LANGUAGE_CODES.items()):
                print("{column_1}{column_2}".format(
                    column_1=lang_code_utils.wjust(code, 18),
                    column_2=language))
        else:
            print(_("Match py-googletrans lang codes."))
            lang_code_utils.match_print(
                dsr_lang=args.list_translation_codes,
                match_list=list(constants.TRANSLATION_LANGUAGE_CODES.keys()),
                min_score=args.min_score)
        return True

    return False


def validate_io(  # pylint: disable=too-many-branches, too-many-statements
        args,
        styles_list):
    """
    Give args and choose workflow depends on the io options.
    """
    if not args.input or not os.path.isfile(args.input):
        raise exceptions.AutosubException(
            _("Error: arg of \"-i\"/\"--input\": \"{path}\" isn't valid. "
              "You need to give a valid path.").format(path=args.input))

    if args.styles:  # pylint: disable=too-many-nested-blocks
        if not os.path.isfile(args.styles):
            raise exceptions.AutosubException(
                _("Error: arg of \"-sty\"/\"--styles\": \"{path}\" isn't valid. "
                  "You need to give a valid path.").format(path=args.styles))

        if args.styles_name:
            if len(args.styles_name) > 2:
                raise exceptions.AutosubException(
                    _("Error: Too many \"-sn\"/\"--styles-name\" arguments."))

            style_obj = pysubs2.SSAFile.load(args.styles)
            ass_styles = style_obj.styles.get(args.styles_name[0])
            if ass_styles:
                styles_dict = {args.styles_name[0]: ass_styles}
                if len(args.styles_name) == 2:
                    ass_styles = style_obj.styles.get(args.styles_name[1])
                    if ass_styles:
                        styles_dict[args.styles_name[1]] = ass_styles
                    else:
                        raise exceptions.AutosubException(
                            _("Error: \"-sn\"/\"--styles-name\" "
                              "arguments aren't in \"{path}\".").format(path=args.styles))
                for item in styles_dict.items():
                    styles_list.append(item[0])
                    styles_list.append(item[1])
            else:
                raise exceptions.AutosubException(
                    _("Error: \"-sn\"/\"--styles-name\" "
                      "arguments aren't in \"{path}\".").format(path=args.styles))

    if args.ext_regions and not os.path.isfile(args.ext_regions):
        raise exceptions.AutosubException(
            _("Error: arg of \"-er\"/\"--ext-regions\": \"{path}\" isn't valid. "
              "You need to give a valid path.").format(path=args.ext_regions))

    input_name = os.path.splitext(args.input)
    input_ext = input_name[-1]
    input_fmt = input_ext.strip('.')
    input_path = input_name[0]

    is_ass_input = input_fmt in constants.INPUT_FORMAT

    if not args.output:
        args.output = input_path
        # get output name from input path
        if not args.format:
            # get format from input
            if is_ass_input:
                args.format = input_fmt
                print(_("No output format specified. "
                        "Use input format \"{fmt}\" "
                        "for output.").format(fmt=input_fmt))
            else:
                # get format from default
                args.format = constants.DEFAULT_SUBTITLES_FORMAT

    elif os.path.isdir(args.output):
        args.output = os.path.join(
            args.output,
            os.path.basename(args.input).rstrip(input_ext))
        # output = path + basename of input without extension
        print(_("Your output is a directory not a file path. "
                "Now file path set to \"{new}\".").format(new=args.output))
        if not args.format:
            # get format from input
            if is_ass_input:
                args.format = input_fmt
                print(_("No output format specified. "
                        "Use input format \"{fmt}\" "
                        "for output.").format(fmt=input_fmt))
            else:
                # get format from default
                args.format = constants.DEFAULT_SUBTITLES_FORMAT
    else:
        output_name = os.path.splitext(args.output)
        if not args.format:
            # get format from output
            args.format = output_name[-1].strip('.')
            # format = output name extension without dot
        args.output = output_name[0]
        # output = output name without extension

    if args.format not in constants.OUTPUT_FORMAT:
        raise exceptions.AutosubException(
            _("Error: Output subtitles format \"{fmt}\" not supported. "
              "Run with \"-lf\"/\"--list-formats\" to see all supported formats.\n"
              "Or use ffmpeg or SubtitleEdit to convert the formats.").format(fmt=args.format))

    args.output_files = set(args.output_files)
    if "all" in args.output_files:
        args.output_files = constants.DEFAULT_MODE_SET
    else:
        if not is_ass_input:
            args.output_files = args.output_files & \
                                constants.DEFAULT_MODE_SET
            if not args.output_files:
                raise exceptions.AutosubException(
                    _("Error: No valid \"-of\"/\"--output-files\" arguments."))

    if args.best_match:
        args.best_match = {k.lower() for k in args.best_match}
        if 'all' in args.best_match:
            args.best_match = constants.DEFAULT_LANG_MODE_SET
        else:
            args.best_match = \
                args.best_match & constants.DEFAULT_LANG_MODE_SET

    if is_ass_input:
        print(_("Input is a subtitles file."))
        return 0

    return 1


def validate_config_args(args):  # pylint: disable=too-many-branches, too-many-return-statements, too-many-statements
    """
    Check that the speech-config args passed to autosub are valid
    for audio or video processing.
    """
    if os.path.isfile(args.speech_config):
        with open(args.speech_config, encoding='utf-8') as config_file:
            try:
                config_dict = json.load(config_file)
            except ValueError:
                raise exceptions.AutosubException(
                    _("Error: Can't decode speech config file \"{filename}\".").format(
                        filename=args.speech_config))
    else:
        raise exceptions.AutosubException(
            _("Error: Speech config file \"{filename}\" doesn't exist.").format(
                filename=args.speech_config))

    if args.speech_api == "gcsv1":
        if "encoding" in config_dict and config_dict["encoding"]:
            # https://cloud.google.com/speech-to-text/docs/quickstart-protocol
            # https://cloud.google.com/speech-to-text/docs/reference/rest/v1p1beta1/RecognitionConfig?hl=zh-cn#AudioEncoding
            args.api_suffix = api_google.google_enc_to_ext(config_dict["encoding"])
        else:
            # it's necessary to set default encoding
            config_dict["encoding"] = api_google.google_ext_to_enc(args.api_suffix)

        # https://cloud.google.com/speech-to-text/docs/reference/rest/v1p1beta1/RecognitionConfig
        # https://googleapis.dev/python/speech/latest/gapic/v1/types.html#google.cloud.speech_v1.types.RecognitionConfig
        # In practice, the client API only accept the Snake case naming variable
        # but the URL API accept the both
        if "sample_rate_hertz" in config_dict and config_dict["sample_rate_hertz"]:
            args.api_sample_rate = config_dict["sample_rate_hertz"]
        elif "sampleRateHertz" in config_dict and config_dict["sampleRateHertz"]:
            args.api_sample_rate = config_dict["sampleRateHertz"]
        else:
            # it's necessary to set sample_rate_hertz from option --api-sample-rate
            config_dict["sample_rate_hertz"] = args.api_sample_rate

        if "audio_channel_count" in config_dict and config_dict["audio_channel_count"]:
            args.api_audio_channel = config_dict["audio_channel_count"]
        elif "audioChannelCount" in config_dict and config_dict["audioChannelCount"]:
            args.api_audio_channel = config_dict["audioChannelCount"]

        if "language_code" in config_dict and config_dict["language_code"]:
            args.speech_language = config_dict["language_code"]
        elif "languageCode" in config_dict and config_dict["languageCode"]:
            args.speech_language = config_dict["languageCode"]

    else:
        args.api_suffix = ".pcm"
        args.api_sample_rate = 16000

        if "APPID" in config_dict:
            config_dict["app_id"] = config_dict["APPID"]
            del config_dict["APPID"]
        elif "AppID" in config_dict:
            config_dict["app_id"] = config_dict["AppID"]
            del config_dict["AppID"]
        elif "app_id" not in config_dict:
            raise exceptions.AutosubException(
                _("Error: No \"app_id\" found in speech config file \"{filename}\"."
                  ).format(filename=args.speech_config))

        if "APIKey" in config_dict:
            config_dict["api_key"] = config_dict["APIKey"]
            del config_dict["APIKey"]
        elif "API key" in config_dict:
            config_dict["api_key"] = config_dict["API key"]
            del config_dict["API key"]
        elif "api_key" not in config_dict:
            raise exceptions.AutosubException(
                _("Error: No \"api_key\" found in speech config file \"{filename}\"."
                  ).format(filename=args.speech_config))

        if "APISecret" in config_dict:
            config_dict["api_secret"] = config_dict["APISecret"]
            del config_dict["APISecret"]
        elif "Secret Key" in config_dict:
            config_dict["api_secret"] = config_dict["Secret Key"]
            del config_dict["Secret Key"]
        elif "api_secret" not in config_dict:
            raise exceptions.AutosubException(
                _("Error: No \"api_secret\" found in speech config file \"{filename}\"."
                  ).format(filename=args.speech_config))

        if args.speech_api == "xfyun":
            if "business" not in config_dict:
                config_dict["business"] = {
                    "language": "zh_cn",
                    "domain": "iat",
                    "accent": "mandarin"}

            if "language" not in config_dict["business"]:
                raise exceptions.AutosubException(
                    _("Error: No \"language\" found in speech config file \"{filename}\"."
                      ).format(filename=args.speech_config))

            args.speech_language = config_dict["business"]["language"]

        elif args.speech_api == "baidu":
            if "config" not in config_dict:
                config_dict["config"] = {
                    "format": "pcm",
                    "rate": 16000,
                    "channel": 1,
                    "cuid": "python",
                    "dev_pid": 1537
                }

            if "dev_pid" in config_dict["config"]:
                args.speech_language = \
                    api_baidu.baidu_dev_pid_to_lang_code(config_dict["config"]["dev_pid"])
            else:
                config_dict["config"]["dev_pid"] = 1537

            if "disable_qps_limit" not in config_dict \
                    or config_dict["disable_qps_limit"] is not True:
                # Queries per second limit
                args.speech_concurrency = 1

    args.speech_config = config_dict


def validate_aovp_args(args):  # pylint: disable=too-many-branches, too-many-return-statements, too-many-statements
    """
    Check that the commandline arguments passed to autosub are valid
    for audio or video processing.
    """
    if args.sleep_seconds < 0:
        raise exceptions.AutosubException(
            _("Error: \"-slp\"/\"--sleep-seconds\" arg is illegal."))

    if args.speech_language:  # pylint: disable=too-many-nested-blocks
        if args.speech_api == "gsv2" or args.speech_api == "gcsv1":
            args.speech_language = args.speech_language.lower()
            if args.speech_language \
                    not in constants.SPEECH_TO_TEXT_LANGUAGE_CODES:
                if args.best_match and 's' in args.best_match:
                    print(_("Let speech lang code to match Google Speech-to-Text lang codes."))
                    best_result = lang_code_utils.match_print(
                        dsr_lang=args.speech_language,
                        match_list=list(constants.SPEECH_TO_TEXT_LANGUAGE_CODES.keys()),
                        min_score=args.min_score)
                    if best_result:
                        print(_("Use langcodes to standardize the result."))
                        args.speech_language = langcodes.standardize_tag(best_result[0])
                        print(_("Use \"{lang_code}\" instead.").format(
                            lang_code=args.speech_language))
                    else:
                        print(_("Match failed. Still using \"{lang_code}\".").format(
                            lang_code=args.speech_language))
                else:
                    print(_("Warning: Speech language \"{src}\" is not recommended. "
                            "Run with \"-lsc\"/\"--list-speech-codes\" "
                            "to see all supported languages. "
                            "Or use \"-bm\"/\"--best-match\" to get a best match."
                           ).format(src=args.speech_language))

            if args.min_confidence < 0.0 or args.min_confidence > 1.0:
                raise exceptions.AutosubException(
                    _("Error: The arg of \"-mnc\"/\"--min-confidence\" isn't legal."))

        elif args.speech_api == "xfyun":
            if not args.speech_config:
                raise exceptions.AutosubException(
                    _("Error: You must provide \"-sconf\", \"--speech-config\" option "
                      "when using Xun Fei Yun API."))

        if args.dst_language is None:
            print(_("Translation destination language not provided. "
                    "Only performing speech recognition."))

        else:
            if not args.src_language:
                print(_("Translation source language not provided. "
                        "Use speech language instead."))
                args.src_language = args.speech_language
                if not args.best_match:
                    args.best_match = {'src'}
                elif 'src' not in args.best_match:
                    args.best_match.add('src')

            is_src_matched = False
            is_dst_matched = False

            for key in googletrans.constants.LANGUAGES:
                if args.src_language.lower() == key.lower():
                    args.src_language = key
                    is_src_matched = True
                if args.dst_language.lower() == key.lower():
                    args.dst_language = key
                    is_dst_matched = True

            if not is_src_matched:
                if args.best_match and 'src' in args.best_match:
                    print(_("Let translation source lang code "
                            "to match py-googletrans lang codes."))
                    best_result = lang_code_utils.match_print(
                        dsr_lang=args.src_language,
                        match_list=list(googletrans.constants.LANGUAGES.keys()),
                        min_score=args.min_score)
                    if best_result:
                        print(_("Use \"{lang_code}\" instead.").format(
                            lang_code=best_result[0]))
                        args.src_language = best_result[0]
                    else:
                        raise exceptions.AutosubException(_("Error: Match failed."))
                else:
                    raise exceptions.AutosubException(
                        _("Error: Translation source language \"{src}\" is not supported. "
                          "Run with \"-ltc\"/\"--list-translation-codes\" "
                          "to see all supported languages. "
                          "Or use \"-bm\"/\"--best-match\" to get a best match.").format(
                              src=args.src_language))

            if not is_dst_matched:
                if args.best_match and 'd' in args.best_match:
                    print(_("Let translation destination lang code "
                            "to match py-googletrans lang codes."))
                    best_result = lang_code_utils.match_print(
                        dsr_lang=args.dst_language,
                        match_list=list(googletrans.constants.LANGUAGES.keys()),
                        min_score=args.min_score)
                    if best_result:
                        print(_("Use \"{lang_code}\" instead.").format(
                            lang_code=best_result[0]))
                        args.dst_language = best_result[0]
                    else:
                        raise exceptions.AutosubException(_("Error: Match failed."))
                else:
                    raise exceptions.AutosubException(
                        _("Error: Translation destination language \"{dst}\" is not supported. "
                          "Run with \"-ltc\"/\"--list-translation-codes\" "
                          "to see all supported languages. "
                          "Or use \"-bm\"/\"--best-match\" to get a best match.").format(
                              dst=args.dst_language))

        if args.dst_language == args.speech_language \
                or args.src_language == args.dst_language:
            print(_("Speech language is the same as the destination language. "
                    "Only performing speech recognition."))
            args.dst_language = None
            args.src_language = None

    else:
        if args.ext_regions:
            if not args.keep:
                raise exceptions.AutosubException(
                    _("You've already input times. "
                      "No works done."))

        else:
            print(_("Speech language not provided. "
                    "Only performing speech regions detection."))

    if args.styles == ' ':
        # when args.styles is used but without option
        # its value is ' '
        if not args.ext_regions:
            raise exceptions.AutosubException(
                _("Error: External speech regions file not provided."))

        args.styles = args.ext_regions


def validate_sp_args(args):  # pylint: disable=too-many-branches,too-many-return-statements, too-many-statements
    """
    Check that the commandline arguments passed to autosub are valid
    for subtitles processing.
    """
    if args.src_language:
        if args.dst_language is None:
            raise exceptions.AutosubException(
                _("Error: Destination language not provided."))

        is_src_matched = False
        is_dst_matched = False

        for key in googletrans.constants.LANGUAGES:
            if args.src_language.lower() == key.lower():
                args.src_language = key
                is_src_matched = True
            if args.dst_language.lower() == key.lower():
                args.dst_language = key
                is_dst_matched = True

        if not is_src_matched:
            if args.best_match and 'src' in args.best_match:
                print(
                    _("Warning: Source language \"{src}\" not supported. "
                      "Run with \"-lsc\"/\"--list-translation-codes\" "
                      "to see all supported languages.").format(src=args.src_language))
                best_result = lang_code_utils.match_print(
                    dsr_lang=args.src_language,
                    match_list=list(googletrans.constants.LANGUAGES.keys()),
                    min_score=args.min_score)
                if best_result:
                    print(_("Use \"{lang_code}\" instead.").format(lang_code=best_result[0]))
                    args.src_language = best_result[0]
                else:
                    raise exceptions.AutosubException(
                        _("Match failed. Still using \"{lang_code}\". "
                          "Program stopped.").format(
                              lang_code=args.src_language))

            else:
                raise exceptions.AutosubException(
                    _("Error: Source language \"{src}\" not supported. "
                      "Run with \"-lsc\"/\"--list-translation-codes\" "
                      "to see all supported languages. "
                      "Or use \"-bm\"/\"--best-match\" to get a best match.").format(
                          src=args.src_language))

        if not is_dst_matched:
            if args.best_match and 'd' in args.best_match:
                print(
                    _("Warning: Destination language \"{dst}\" not supported. "
                      "Run with \"-lsc\"/\"--list-translation-codes\" "
                      "to see all supported languages.").format(dst=args.dst_language))
                best_result = lang_code_utils.match_print(
                    dsr_lang=args.dst_language,
                    match_list=list(googletrans.constants.LANGUAGES.keys()),
                    min_score=args.min_score)
                if best_result:
                    print(_("Use \"{lang_code}\" instead.").format(lang_code=best_result[0]))
                    args.dst_language = best_result[0]
                else:
                    raise exceptions.AutosubException(
                        _("Match failed. Still using \"{lang_code}\". "
                          "Program stopped.").format(
                              lang_code=args.dst_language))

            else:
                raise exceptions.AutosubException(
                    _("Error: Destination language \"{dst}\" not supported. "
                      "Run with \"-lsc\"/\"--list-translation-codes\" "
                      "to see all supported languages. "
                      "Or use \"-bm\"/\"--best-match\" to get a best match.").format(
                          dst=args.dst_language))

        if args.dst_language == args.src_language:
            raise exceptions.AutosubException(
                _("Error: Translation source language is the same as the destination language."))

    else:
        return 0

    if args.styles == ' ':
        # when args.styles is used but without option
        # its value is ' '
        if not args.ext_regions:
            raise exceptions.AutosubException(
                _("Error: External speech regions file not provided."))

        args.styles = args.ext_regions

    return 1


def fix_args(args):
    """
    Check that the commandline arguments value passed to autosub are proper.
    """
    if not args.ext_regions:
        if args.min_region_size < constants.MIN_REGION_SIZE_LIMIT:
            print(
                _("Your minimum region size {mrs0} is smaller than {mrs}.\n"
                  "Now reset to {mrs}.").format(mrs0=args.min_region_size,
                                                mrs=constants.MIN_REGION_SIZE_LIMIT))
            args.min_region_size = constants.MIN_REGION_SIZE_LIMIT

        if args.max_region_size > constants.MAX_REGION_SIZE_LIMIT:
            print(
                _("Your maximum region size {mrs0} is larger than {mrs}.\n"
                  "Now reset to {mrs}.").format(mrs0=args.max_region_size,
                                                mrs=constants.MAX_REGION_SIZE_LIMIT))
            args.max_region_size = constants.MAX_REGION_SIZE_LIMIT

        if args.max_continuous_silence < 0:
            print(
                _("Your maximum continuous silence {mxcs} is smaller than 0.\n"
                  "Now reset to {dmxcs}.").format(mxcs=args.max_continuous_silence,
                                                  dmxcs=constants.DEFAULT_CONTINUOUS_SILENCE))
            args.max_continuous_silence = constants.DEFAULT_CONTINUOUS_SILENCE


def get_timed_text(
        is_empty_dropped,
        regions,
        text_list):
    """
    Get timed text list.
    """
    if is_empty_dropped:
        # drop empty regions
        timed_text = [(region, text) for region, text in zip(regions, text_list) if text]
    else:
        # keep empty regions
        timed_text = list(zip(regions, text_list))

    return timed_text


def sub_conversion(  # pylint: disable=too-many-branches, too-many-statements, too-many-locals
        args,
        input_m=input,
        fps=30.0):
    """
    Give args and convert a subtitles file.
    """
    src_sub = pysubs2.SSAFile.load(args.input)
    try:
        args.output_files.remove("dst-lf-src")
        new_sub = sub_utils.merge_bilingual_assfile(
            subtitles=src_sub
        )
        sub_string = core.ssafile_to_sub_str(
            ssafile=new_sub,
            fps=fps,
            subtitles_file_format=args.format)

        if args.format == 'mpl2':
            extension = 'mpl2.txt'
        else:
            extension = args.format

        sub_name = "{base}.{nt}.{extension}".format(
            base=args.output,
            nt="combination",
            extension=extension)

        subtitles_file_path = core.str_to_file(
            str_=sub_string,
            output=sub_name,
            input_m=input_m)
        # subtitles string to file
        print(_("\"dst-lf-src\" subtitles file "
                "created at \"{}\".").format(subtitles_file_path))

        if not args.output_files:
            raise exceptions.AutosubException(_("\nAll works done."))

    except KeyError:
        pass

    try:
        args.output_files.remove("src-lf-dst")
        new_sub = sub_utils.merge_bilingual_assfile(
            subtitles=src_sub,
            order=0
        )
        sub_string = core.ssafile_to_sub_str(
            ssafile=new_sub,
            fps=fps,
            subtitles_file_format=args.format)

        if args.format == 'mpl2':
            extension = 'mpl2.txt'
        else:
            extension = args.format

        sub_name = "{base}.{nt}.{extension}".format(
            base=args.output,
            nt="combination.2",
            extension=extension)

        subtitles_file_path = core.str_to_file(
            str_=sub_string,
            output=sub_name,
            input_m=input_m)
        # subtitles string to file
        print(_("\"src-lf-dst\" subtitles file "
                "created at \"{}\".").format(subtitles_file_path))

        if not args.output_files:
            raise exceptions.AutosubException(_("\nAll works done."))

    except KeyError:
        pass

    try:
        args.output_files.remove("join-events")
        if args.stop_words_1:
            stop_words_1 = args.stop_words_1.split(" ")
            stop_words_set_1 = set(stop_words_1)
        else:
            stop_words_set_1 = constants.DEFAULT_ENGLISH_STOP_WORDS_SET_1
        if args.stop_words_2:
            stop_words_2 = args.stop_words_2.split(" ")
            stop_words_set_2 = set(stop_words_2)
        else:
            stop_words_set_2 = constants.DEFAULT_ENGLISH_STOP_WORDS_SET_2

        new_sub = sub_utils.merge_src_assfile(
            subtitles=src_sub,
            max_join_size=args.max_join_size,
            max_delta_time=int(args.max_delta_time * 1000),
            delimiters=args.delimiters,
            stop_words_set_1=stop_words_set_1,
            stop_words_set_2=stop_words_set_2,
            avoid_split=args.dont_split
        )
        sub_string = core.ssafile_to_sub_str(
            ssafile=new_sub,
            fps=fps,
            subtitles_file_format=args.format)

        if args.format == 'mpl2':
            extension = 'mpl2.txt'
        else:
            extension = args.format

        sub_name = "{base}.{nt}.{extension}".format(
            base=args.output,
            nt="join",
            extension=extension)

        subtitles_file_path = core.str_to_file(
            str_=sub_string,
            output=sub_name,
            input_m=input_m)
        # subtitles string to file
        print(_("\"join-events\" subtitles file "
                "created at \"{}\".").format(subtitles_file_path))

        if not args.output_files:
            raise exceptions.AutosubException(_("\nAll works done."))

    except KeyError:
        pass


def sub_trans(  # pylint: disable=too-many-branches, too-many-statements, too-many-locals
        args,
        input_m=input,
        fps=30.0,
        styles_list=None):
    """
    Give args and translate a subtitles file.
    """
    src_sub = pysubs2.SSAFile.load(args.input)
    text_list = []

    if args.styles and \
            (args.format == 'ass' or
             args.format == 'ssa' or
             args.format == 'ass.json'):
        src_sub.styles = \
            {styles_list[i]: styles_list[i + 1] for i in range(0, len(styles_list), 2)}
        for event in src_sub.events:
            event.style = styles_list[0]
            text_list.append(event.text)
    else:
        styles_list = [src_sub.events[0].style, ]
        for event in src_sub.events:
            text_list.append(event.text)

    # text translation
    # use googletrans
    translated_text = core.list_to_googletrans(
        text_list,
        src_language=args.src_language,
        dst_language=args.dst_language,
        sleep_seconds=args.sleep_seconds,
        user_agent=args.user_agent,
        service_urls=args.service_urls,
        drop_override_codes=args.drop_override_codes,
        delete_chars=args.gt_delete_chars)

    if not translated_text or len(translated_text) != len(text_list):
        raise exceptions.AutosubException(
            _("Error: Translation failed."))

    try:
        args.output_files.remove("bilingual")
        bilingual_sub = pysubs2.SSAFile()
        bilingual_sub.styles = src_sub.styles
        bilingual_sub.info = src_sub.info
        bilingual_sub.events = src_sub.events[:]
        if args.styles and \
                len(styles_list) == 2 and \
                (args.format == 'ass' or
                 args.format == 'ssa' or
                 args.format == 'ass.json'):
            sub_utils.pysubs2_ssa_event_add(
                src_ssafile=bilingual_sub,
                dst_ssafile=bilingual_sub,
                text_list=translated_text,
                style_name=styles_list[2])
        else:
            sub_utils.pysubs2_ssa_event_add(
                src_ssafile=bilingual_sub,
                dst_ssafile=bilingual_sub,
                text_list=translated_text,
                style_name="")

        bilingual_string = core.ssafile_to_sub_str(
            ssafile=bilingual_sub,
            fps=fps,
            subtitles_file_format=args.format)

        if args.format == 'mpl2':
            extension = 'mpl2.txt'
        else:
            extension = args.format

        bilingual_name = "{base}.{nt}.{extension}".format(
            base=args.output,
            nt=args.src_language + '&' + args.dst_language,
            extension=extension)

        subtitles_file_path = core.str_to_file(
            str_=bilingual_string,
            output=bilingual_name,
            input_m=input_m)
        # subtitles string to file
        print(_("Bilingual subtitles file "
                "created at \"{}\".").format(subtitles_file_path))

        if not args.output_files:
            raise exceptions.AutosubException(_("\nAll works done."))

    except KeyError:
        pass

    try:
        args.output_files.remove("dst-lf-src")
        bilingual_sub = pysubs2.SSAFile()
        bilingual_sub.styles = src_sub.styles
        bilingual_sub.info = src_sub.info
        if args.styles and \
                len(styles_list) == 2 and \
                (args.format == 'ass' or
                 args.format == 'ssa' or
                 args.format == 'ass.json'):
            sub_utils.pysubs2_ssa_event_add(
                src_ssafile=src_sub,
                dst_ssafile=bilingual_sub,
                text_list=translated_text,
                style_name=styles_list[2],
                same_event_type=1)
        else:
            sub_utils.pysubs2_ssa_event_add(
                src_ssafile=src_sub,
                dst_ssafile=bilingual_sub,
                text_list=translated_text,
                style_name="",
                same_event_type=1)

        bilingual_string = core.ssafile_to_sub_str(
            ssafile=bilingual_sub,
            fps=fps,
            subtitles_file_format=args.format)

        if args.format == 'mpl2':
            extension = 'mpl2.txt'
        else:
            extension = args.format

        bilingual_name = "{base}.{nt}.0.{extension}".format(
            base=args.output,
            nt=args.src_language + '&' + args.dst_language,
            extension=extension)

        subtitles_file_path = core.str_to_file(
            str_=bilingual_string,
            output=bilingual_name,
            input_m=input_m)
        # subtitles string to file
        print(_("\"dst-lf-src\" subtitles file "
                "created at \"{}\".").format(subtitles_file_path))

        if not args.output_files:
            raise exceptions.AutosubException(_("\nAll works done."))

    except KeyError:
        pass

    try:
        args.output_files.remove("src-lf-dst")
        bilingual_sub = pysubs2.SSAFile()
        bilingual_sub.styles = src_sub.styles
        bilingual_sub.info = src_sub.info
        if args.styles and \
                len(styles_list) == 2 and \
                (args.format == 'ass' or
                 args.format == 'ssa' or
                 args.format == 'ass.json'):
            sub_utils.pysubs2_ssa_event_add(
                src_ssafile=src_sub,
                dst_ssafile=bilingual_sub,
                text_list=translated_text,
                style_name=styles_list[2],
                same_event_type=2)
        else:
            sub_utils.pysubs2_ssa_event_add(
                src_ssafile=src_sub,
                dst_ssafile=bilingual_sub,
                text_list=translated_text,
                style_name="",
                same_event_type=2)

        bilingual_string = core.ssafile_to_sub_str(
            ssafile=bilingual_sub,
            fps=fps,
            subtitles_file_format=args.format)

        if args.format == 'mpl2':
            extension = 'mpl2.txt'
        else:
            extension = args.format

        bilingual_name = "{base}.{nt}.1.{extension}".format(
            base=args.output,
            nt=args.src_language + '&' + args.dst_language,
            extension=extension)

        subtitles_file_path = core.str_to_file(
            str_=bilingual_string,
            output=bilingual_name,
            input_m=input_m)
        # subtitles string to file
        print(_("\"src-lf-dst\" subtitles file "
                "created at \"{}\".").format(subtitles_file_path))

        if not args.output_files:
            raise exceptions.AutosubException(_("\nAll works done."))

    except KeyError:
        pass

    try:
        args.output_files.remove("dst")
        dst_sub = pysubs2.SSAFile()
        dst_sub.styles = src_sub.styles
        dst_sub.info = src_sub.info
        if len(styles_list) == 2:
            sub_utils.pysubs2_ssa_event_add(
                src_ssafile=src_sub,
                dst_ssafile=dst_sub,
                text_list=translated_text,
                style_name=styles_list[2])
        else:
            sub_utils.pysubs2_ssa_event_add(
                src_ssafile=src_sub,
                dst_ssafile=dst_sub,
                text_list=translated_text,
                style_name="")

        dst_string = core.ssafile_to_sub_str(
            ssafile=dst_sub,
            fps=fps,
            subtitles_file_format=args.format)

        if args.format == 'mpl2':
            extension = 'mpl2.txt'
        else:
            extension = args.format
        dst_name = "{base}.{nt}.{extension}".format(base=args.output,
                                                    nt=args.dst_language,
                                                    extension=extension)
        subtitles_file_path = core.str_to_file(
            str_=dst_string,
            output=dst_name,
            input_m=input_m)
        # subtitles string to file
        print(_("Destination language subtitles "
                "file created at \"{}\".").format(subtitles_file_path))

    except KeyError:
        pass


def get_fps(
        args,
        input_m=input):
    """
    Give args and get fps.
    """
    if args.format == 'sub':
        if not args.sub_fps:
            fps = ffmpeg_utils.ffprobe_get_fps(
                args.input,
                input_m=input_m)
            if fps == 0.0:
                if not args.yes:
                    args.format = 'srt'
                else:
                    raise pysubs2.exceptions.Pysubs2Error
        else:
            fps = args.sub_fps
    else:
        fps = 0.0

    return fps


def audio_or_video_prcs(  # pylint: disable=too-many-branches, too-many-statements, too-many-locals, too-many-arguments
        args,
        input_m=input,
        fps=30.0,
        styles_list=None):
    """
    Give args and process an input audio or video file.
    """
    if args.ext_regions:
        # use external speech regions
        print(_("Use external speech regions."))
        audio_wav_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        audio_wav = audio_wav_temp.name
        audio_wav_temp.close()
        command = args.audio_conversion_cmd.format(
            in_=args.input,
            channel=1,
            sample_rate=16000,
            out_=audio_wav)
        print(command)
        prcs = subprocess.Popen(constants.cmd_conversion(command),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        out, err = prcs.communicate()
        if out:
            print(out.decode(sys.stdout.encoding))
        if err:
            print(err.decode(sys.stdout.encoding))
        regions = sub_utils.sub_to_speech_regions(
            audio_wav=audio_wav,
            sub_file=args.ext_regions)
        os.remove(audio_wav)

    else:
        # use auditok_gen_speech_regions
        mode = 0
        if not args.not_strict_min_length:
            mode = auditok.StreamTokenizer.STRICT_MIN_LENGTH
        if args.drop_trailing_silence:
            mode = mode | auditok.StreamTokenizer.DROP_TRAILING_SILENCE

        audio_wav_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        audio_wav = audio_wav_temp.name
        audio_wav_temp.close()
        command = args.audio_conversion_cmd.format(
            in_=args.input,
            channel=1,
            sample_rate=48000,
            out_=audio_wav)
        print(_("\nConvert source file to \"{name}\" "
                "to detect audio regions.").format(
                    name=audio_wav))
        print(command)
        prcs = subprocess.Popen(constants.cmd_conversion(command),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        out, err = prcs.communicate()
        if out:
            print(out.decode(sys.stdout.encoding))
        if err:
            print(err.decode(sys.stdout.encoding))

        if not ffmpeg_utils.ffprobe_check_file(audio_wav):
            raise exceptions.AutosubException(
                _("Error: Convert source file to \"{name}\" failed.").format(
                    name=audio_wav))

        print(_("Conversion completed.\nUse Auditok to detect speech regions."))

        regions = core.auditok_gen_speech_regions(
            audio_wav=audio_wav,
            energy_threshold=args.energy_threshold,
            min_region_size=args.min_region_size,
            max_region_size=args.max_region_size,
            max_continuous_silence=args.max_continuous_silence,
            mode=mode)
        os.remove(audio_wav)
        gc.collect(0)

        print(_("Auditok detection completed."
                "\n\"{name}\" has been deleted.").format(name=audio_wav))

    if not regions:
        raise exceptions.AutosubException(
            _("Error: Can't get speech regions."))
    if args.speech_language or \
            args.audio_process and 's' in args.audio_process:
        # process output first
        try:
            args.output_files.remove("regions")
            if args.styles and \
                    (args.format == 'ass' or
                     args.format == 'ssa' or
                     args.format == 'ass.json'):
                times_string = core.list_to_ass_str(
                    text_list=regions,
                    styles_list=styles_list,
                    subtitles_file_format=args.format)
            else:
                times_string = core.list_to_sub_str(
                    timed_text=regions,
                    fps=fps,
                    subtitles_file_format=args.format)
            # times to subtitles string
            times_name = "{base}.{nt}.{extension}".format(base=args.output,
                                                          nt="times",
                                                          extension=args.format)
            subtitles_file_path = core.str_to_file(
                str_=times_string,
                output=times_name,
                input_m=input_m)
            # subtitles string to file

            print(_("Times file created at \"{}\".").format(subtitles_file_path))

            if not args.output_files:
                raise exceptions.AutosubException(_("\nAll works done."))

        except KeyError:
            pass

        audio_fragments = core.bulk_audio_conversion(
            source_file=args.input,
            output=args.output,
            regions=regions,
            split_cmd=args.audio_split_cmd,
            suffix=args.api_suffix,
            concurrency=args.audio_concurrency,
            is_keep=args.keep)
        gc.collect(0)

        if not audio_fragments or \
                len(audio_fragments) != len(regions):
            if not args.keep:
                for audio_fragment in audio_fragments:
                    os.remove(audio_fragment)
            raise exceptions.ConversionException(
                _("Error: Conversion failed."))

        if args.audio_process and 's' in args.audio_process:
            raise exceptions.AutosubException(
                _("Audio processing complete.\nAll works done."))

        try:
            args.output_files.remove("full-src")
            result_list = []
        except KeyError:
            result_list = None

        if args.speech_api == "gsv2":
            # Google speech-to-text v2
            if args.http_speech_api:
                gsv2_api_url = "http://" + \
                               constants.GOOGLE_SPEECH_V2_API_URL
            else:
                gsv2_api_url = "https://" + \
                               constants.GOOGLE_SPEECH_V2_API_URL

            if args.speech_key:
                gsv2_api_url = gsv2_api_url.format(
                    lang=args.speech_language,
                    key=args.speech_key)
            else:
                gsv2_api_url = gsv2_api_url.format(
                    lang=args.speech_language,
                    key=constants.GOOGLE_SPEECH_V2_API_KEY)

            if args.api_suffix == ".flac":
                headers = \
                    {"Content-Type": "audio/x-flac; rate={rate}".format(rate=args.api_sample_rate)}
            else:
                headers = \
                    {"Content-Type": "audio/ogg; rate={rate}".format(rate=args.api_sample_rate)}

            text_list = core.gsv2_to_text(
                audio_fragments=audio_fragments,
                api_url=gsv2_api_url,
                headers=headers,
                concurrency=args.speech_concurrency,
                min_confidence=args.min_confidence,
                is_keep=args.keep,
                result_list=result_list)
            gc.collect(0)

        elif args.speech_api == "gcsv1":
            # Google Cloud speech-to-text V1P1Beta1
            if args.speech_key:
                headers = \
                    {"Content-Type": "application/json"}
                gcsv1_api_url = \
                    "https://speech.googleapis.com/" \
                    "v1p1beta1/speech:recognize?key={api_key}".format(
                        api_key=args.speech_key)
                print(_("Use the API key "
                        "given in the option \"-skey\"/\"--speech-key\"."))
                text_list = core.gcsv1_to_text(
                    audio_fragments=audio_fragments,
                    sample_rate=args.api_sample_rate,
                    api_url=gcsv1_api_url,
                    headers=headers,
                    config=args.speech_config,
                    concurrency=args.speech_concurrency,
                    src_language=args.speech_language,
                    min_confidence=args.min_confidence,
                    is_keep=args.keep,
                    result_list=result_list)
            elif not constants.IS_GOOGLECLOUDCLIENT:
                raise exceptions.SpeechToTextException(
                    _("Error: Current build version doesn't support "
                      "Google Cloud service account credentials."
                      "\nPlease use other build version "
                      "or use option \"-skey\"/\"--speech-key\" instead."))
            elif args.service_account and os.path.isfile(args.service_account):
                print(_("Set the GOOGLE_APPLICATION_CREDENTIALS "
                        "given in the option \"-sa\"/\"--service-account\"."))
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args.service_account
                text_list = core.gcsv1_to_text(
                    audio_fragments=audio_fragments,
                    sample_rate=args.api_sample_rate,
                    config=args.speech_config,
                    concurrency=args.speech_concurrency,
                    src_language=args.speech_language,
                    min_confidence=args.min_confidence,
                    is_keep=args.keep,
                    result_list=result_list)
            else:
                if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
                    print(_("Use the GOOGLE_APPLICATION_CREDENTIALS "
                            "in the environment variables."))
                    text_list = core.gcsv1_to_text(
                        audio_fragments=audio_fragments,
                        sample_rate=args.api_sample_rate,
                        config=args.speech_config,
                        concurrency=args.speech_concurrency,
                        src_language=args.speech_language,
                        min_confidence=args.min_confidence,
                        is_keep=args.keep,
                        result_list=result_list)
                else:
                    print(_("No available GOOGLE_APPLICATION_CREDENTIALS. "
                            "Use \"-sa\"/\"--service-account\" to set one."))
                    text_list = None

        elif args.speech_api == "xfyun":
            # Xun Fei Yun Speech-to-Text WebSocket API
            text_list = core.xfyun_to_text(
                audio_fragments=audio_fragments,
                config=args.speech_config,
                concurrency=args.speech_concurrency,
                is_keep=False,
                result_list=result_list)
        elif args.speech_api == "baidu":
            # Baidu ASR API
            text_list = core.baidu_to_text(
                audio_fragments=audio_fragments,
                config=args.speech_config,
                concurrency=args.speech_concurrency,
                is_keep=False,
                result_list=result_list)
        elif args.speech_api == "deepspeech":
            text_list = core.deepspeech_to_text(
                audio_fragments=audio_fragments,
                result_list=result_list)
        else:
            text_list = None

        gc.collect(0)

        if result_list and result_list is not None:
            timed_result = get_timed_text(
                is_empty_dropped=False,
                regions=regions,
                text_list=result_list)
            result_string = sub_utils.list_to_json_str(timed_result)
            result_name = "{base}.result.json".format(base=args.output)
            result_file_path = core.str_to_file(
                str_=result_string,
                output=result_name,
                input_m=input_m)
            print(_("Speech-to-Text recogntion result json "
                    "file created at \"{}\".").format(result_file_path))

            if not args.output_files:
                raise exceptions.AutosubException(_("\nAll works done."))

        if not text_list or len(text_list) != len(regions):
            raise exceptions.SpeechToTextException(
                _("Error: Speech-to-text failed.\nAll works done."))

        timed_text = get_timed_text(
            is_empty_dropped=args.drop_empty_regions,
            regions=regions,
            text_list=text_list)

        if args.dst_language:
            # process output first
            try:
                args.output_files.remove("src")
                if args.styles and \
                        (args.format == 'ass' or
                         args.format == 'ssa' or
                         args.format == 'ass.json'):
                    src_string = core.list_to_ass_str(
                        text_list=timed_text,
                        styles_list=styles_list[:2],
                        subtitles_file_format=args.format, )
                else:
                    src_string = core.list_to_sub_str(
                        timed_text=timed_text,
                        fps=fps,
                        subtitles_file_format=args.format)

                # formatting timed_text to subtitles string
                src_name = "{base}.{nt}.{extension}".format(base=args.output,
                                                            nt=args.speech_language,
                                                            extension=args.format)
                subtitles_file_path = core.str_to_file(
                    str_=src_string,
                    output=src_name,
                    input_m=input_m)
                # subtitles string to file
                print(_("Speech language subtitles "
                        "file created at \"{}\".").format(subtitles_file_path))

                if not args.output_files:
                    raise exceptions.AutosubException(_("\nAll works done."))

            except KeyError:
                pass

            # text translation
            translated_text = core.list_to_googletrans(
                text_list,
                src_language=args.src_language,
                dst_language=args.dst_language,
                sleep_seconds=args.sleep_seconds,
                user_agent=args.user_agent,
                service_urls=args.service_urls,
                drop_override_codes=args.drop_override_codes,
                delete_chars=args.gt_delete_chars)

            if not translated_text or len(translated_text) != len(regions):
                raise exceptions.AutosubException(
                    _("Error: Translation failed."))

            try:
                args.output_files.remove("bilingual")
                if args.styles and \
                        (args.format == 'ass' or
                         args.format == 'ssa' or
                         args.format == 'ass.json'):
                    bilingual_string = core.list_to_ass_str(
                        text_list=[timed_text, translated_text],
                        styles_list=styles_list,
                        subtitles_file_format=args.format, )
                else:
                    bilingual_sub = pysubs2.SSAFile()
                    sub_utils.pysubs2_ssa_event_add(
                        src_ssafile=None,
                        dst_ssafile=bilingual_sub,
                        text_list=timed_text)
                    sub_utils.pysubs2_ssa_event_add(
                        src_ssafile=bilingual_sub,
                        dst_ssafile=bilingual_sub,
                        text_list=translated_text,
                        same_event_type=0)
                    bilingual_string = core.ssafile_to_sub_str(
                        ssafile=bilingual_sub,
                        fps=fps,
                        subtitles_file_format=args.format)
                # formatting timed_text to subtitles string
                bilingual_name = "{base}.{nt}.{extension}".format(
                    base=args.output,
                    nt=args.src_language + '&' + args.dst_language,
                    extension=args.format)
                subtitles_file_path = core.str_to_file(
                    str_=bilingual_string,
                    output=bilingual_name,
                    input_m=input_m)
                # subtitles string to file
                print(_("Bilingual subtitles file "
                        "created at \"{}\".").format(subtitles_file_path))

                if not args.output_files:
                    raise exceptions.AutosubException(_("\nAll works done."))

            except KeyError:
                pass

            try:
                args.output_files.remove("dst-lf-src")
                if args.styles and \
                        (args.format == 'ass' or
                         args.format == 'ssa' or
                         args.format == 'ass.json'):
                    bilingual_string = core.list_to_ass_str(
                        text_list=[timed_text, translated_text],
                        styles_list=styles_list,
                        subtitles_file_format=args.format,
                        same_event_type=1)
                else:
                    bilingual_sub = pysubs2.SSAFile()
                    src_sub = pysubs2.SSAFile()
                    sub_utils.pysubs2_ssa_event_add(
                        src_ssafile=None,
                        dst_ssafile=src_sub,
                        text_list=timed_text)
                    sub_utils.pysubs2_ssa_event_add(
                        src_ssafile=src_sub,
                        dst_ssafile=bilingual_sub,
                        text_list=translated_text,
                        same_event_type=1)
                    bilingual_string = core.ssafile_to_sub_str(
                        ssafile=bilingual_sub,
                        fps=fps,
                        subtitles_file_format=args.format)
                # formatting timed_text to subtitles string
                bilingual_name = "{base}.{nt}.0.{extension}".format(
                    base=args.output,
                    nt=args.src_language + '&' + args.dst_language,
                    extension=args.format)
                subtitles_file_path = core.str_to_file(
                    str_=bilingual_string,
                    output=bilingual_name,
                    input_m=input_m)
                # subtitles string to file
                print(_("\"dst-lf-src\" subtitles file "
                        "created at \"{}\".").format(subtitles_file_path))

                if not args.output_files:
                    raise exceptions.AutosubException(_("\nAll works done."))

            except KeyError:
                pass

            try:
                args.output_files.remove("src-lf-dst")
                if args.styles and \
                        (args.format == 'ass' or
                         args.format == 'ssa' or
                         args.format == 'ass.json'):
                    bilingual_string = core.list_to_ass_str(
                        text_list=[timed_text, translated_text],
                        styles_list=styles_list,
                        subtitles_file_format=args.format,
                        same_event_type=2)
                else:
                    bilingual_sub = pysubs2.SSAFile()
                    src_sub = pysubs2.SSAFile()
                    sub_utils.pysubs2_ssa_event_add(
                        src_ssafile=None,
                        dst_ssafile=src_sub,
                        text_list=timed_text)
                    sub_utils.pysubs2_ssa_event_add(
                        src_ssafile=src_sub,
                        dst_ssafile=bilingual_sub,
                        text_list=translated_text,
                        same_event_type=2)
                    bilingual_string = core.ssafile_to_sub_str(
                        ssafile=bilingual_sub,
                        fps=fps,
                        subtitles_file_format=args.format)
                # formatting timed_text to subtitles string
                bilingual_name = "{base}.{nt}.1.{extension}".format(
                    base=args.output,
                    nt=args.src_language + '&' + args.dst_language,
                    extension=args.format)
                subtitles_file_path = core.str_to_file(
                    str_=bilingual_string,
                    output=bilingual_name,
                    input_m=input_m)
                # subtitles string to file
                print(_("\"src-lf-dst\" subtitles file "
                        "created at \"{}\".").format(subtitles_file_path))

                if not args.output_files:
                    raise exceptions.AutosubException(_("\nAll works done."))

            except KeyError:
                pass

            try:
                args.output_files.remove("dst")
                timed_trans = get_timed_text(
                    is_empty_dropped=False,
                    regions=regions,
                    text_list=translated_text
                )
                # formatting timed_text to subtitles string
                if args.styles and \
                        (args.format == 'ass' or
                         args.format == 'ssa' or
                         args.format == 'ass.json'):
                    if len(args.styles) == 4:
                        dst_string = core.list_to_ass_str(
                            text_list=timed_trans,
                            styles_list=styles_list[2:4],
                            subtitles_file_format=args.format, )
                    else:
                        dst_string = core.list_to_ass_str(
                            text_list=timed_trans,
                            styles_list=styles_list,
                            subtitles_file_format=args.format, )
                else:
                    dst_string = core.list_to_sub_str(
                        timed_text=timed_trans,
                        fps=fps,
                        subtitles_file_format=args.format)
                dst_name = "{base}.{nt}.{extension}".format(
                    base=args.output,
                    nt=args.dst_language,
                    extension=args.format)
                subtitles_file_path = core.str_to_file(
                    str_=dst_string,
                    output=dst_name,
                    input_m=input_m)
                # subtitles string to file
                print(_("Destination language subtitles "
                        "file created at \"{}\".").format(subtitles_file_path))

            except KeyError:
                pass

        else:
            if len(args.output_files) > 1 or not ({"dst", "src"} & args.output_files):
                print(
                    _("Override \"-of\"/\"--output-files\" due to your args too few."
                      "\nOutput source subtitles file only."))
            timed_text = get_timed_text(
                is_empty_dropped=args.drop_empty_regions,
                regions=regions,
                text_list=text_list)
            if args.styles and \
                    (args.format == 'ass' or
                     args.format == 'ssa' or
                     args.format == 'ass.json'):
                src_string = core.list_to_ass_str(
                    text_list=timed_text,
                    styles_list=styles_list,
                    subtitles_file_format=args.format, )
            else:
                src_string = core.list_to_sub_str(
                    timed_text=timed_text,
                    fps=fps,
                    subtitles_file_format=args.format)
            # formatting timed_text to subtitles string
            src_name = "{base}.{nt}.{extension}".format(base=args.output,
                                                        nt=args.speech_language,
                                                        extension=args.format)
            subtitles_file_path = core.str_to_file(
                str_=src_string,
                output=src_name,
                input_m=input_m)
            # subtitles string to file
            print(_("Speech language subtitles "
                    "file created at \"{}\".").format(subtitles_file_path))

    else:
        print(
            _("Override \"-of\"/\"--output-files\" due to your args too few."
              "\nOutput regions subtitles file only."))
        if args.styles and \
                (args.format == 'ass' or
                 args.format == 'ssa' or
                 args.format == 'ass.json'):
            times_subtitles = core.list_to_ass_str(
                text_list=regions,
                styles_list=styles_list,
                subtitles_file_format=args.format)
        else:
            times_subtitles = core.list_to_sub_str(
                timed_text=regions,
                fps=fps,
                subtitles_file_format=args.format)
        # times to subtitles string
        times_name = "{base}.{nt}.{extension}".format(base=args.output,
                                                      nt="times",
                                                      extension=args.format)
        subtitles_file_path = core.str_to_file(
            str_=times_subtitles,
            output=times_name,
            input_m=input_m)
        # subtitles string to file

        print(_("Times file created at \"{}\".").format(subtitles_file_path))
