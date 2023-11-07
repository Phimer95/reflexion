from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory
from generators.model import Message
from typing import List
import re

from utils import read_jsonl

def generate_requirements_list(func_sig, model):
    messages = [
        Message(
            role="system",
            content="""
            Your goal is to create a numbered list of requirements for the following function instruction. Do not introduce additional information. Only mention explicitly stated information from the instruction.
            """,
        ),
        Message(
            role="user",
            content="\n" + func_sig,
        ),
    ]

    return model.generate_chat(messages=messages, num_comps=1, temperature=0.0)

def generate_test_feedback(test1, test2, requirements_list, model):
    messages = [
        Message(
            role="system",
            content="""
              You will be given a list of requirements for a function and 2 unit tests. Only one of the tests is correct. Your goal is to decide which one. Think step by step.
            """
        ),
        Message(
            role="user",
            content="Requirements:" + requirements_list + "\n" + "Test 1:" + test1 + "\n" + "Test 2:" + test2,
        ),
    ]

    return model.generate_chat(messages=messages, num_comps=1, temperature=0.0)

def generate_test_feedback_binary(feedback, model):
    messages = [
        Message(
            role="system",
            content="""
               Your goal is to find out if the first or second test is correct, considering the given conclusion. Return "first" if the first test is correct, "second" if the second test is correct. Do not include any additional information.
            """,
        ),
        Message(
            role="user",
            content=feedback,
        ),
    ]
    return model.generate_chat(messages=messages, num_comps=1, temperature=0.0)

def categorize_tests(feedback, assert_statements):
    # Initialize lists for passed and failed tests
    passed_tests = []
    failed_tests = []

    # Split the feedback by lines
    feedback_lines = feedback.split('\n')

    # Go through each assert statement and categorize it
    for statement in assert_statements:
        # Check if the statement is in the feedback as a passed test
        if statement in feedback_lines:
            passed_tests.append(statement)
        else:
            # Since it's not in the passed section, it's a failed test
            failed_tests.append(statement)

    return passed_tests, failed_tests

def get_correct_test(assert_statements, item, model):
    requirements_list = generate_requirements_list(item["prompt"], model)
    test_feedback = generate_test_feedback(assert_statements[0], assert_statements[1], requirements_list, model)
    test_feedback_binary = generate_test_feedback_binary(test_feedback, model)
    if "first" in test_feedback_binary.lower():
        return assert_statements[0]
    elif "second" in test_feedback_binary.lower():
        return assert_statements[1]
    else:
        return None

def refine_tests(failed_tests, passed_tests, gen, model, item):
    # use self-reflection to iteratively improve failed unit tests if they are wrong
    has_flawed_tests = False
    for failed_test in failed_tests:
        feedback_for_failed_test = generate_test_feedback(failed_test, item["prompt"], model)
        feedback_for_failed_test_binary = generate_test_feedback_binary(feedback_for_failed_test, model)
        if "false" in feedback_for_failed_test_binary.lower():
            if not has_flawed_tests:
                has_flawed_tests = True
            refined_test = gen.generate_refined_test(item["prompt"], failed_test, feedback_for_failed_test, model)
            passed_tests.append(refined_test)
        elif "true" in feedback_for_failed_test_binary.lower():
            passed_tests.append(failed_test)
            
    return passed_tests, has_flawed_tests

def refine_all_tests(tests_i, gen, model, item):
    # use self-reflection to iteratively improve failed unit tests if they are wrong
    passed_tests = []
    requirements_list = generate_requirements_list(item["prompt"], model)
    has_flawed_tests = False
    for test in tests_i:
        feedback_for_test = generate_test_feedback(test, requirements_list, model)
        feedback_for_test_binary = generate_test_feedback_binary(feedback_for_test, model)
        if "false" in feedback_for_test_binary.lower():
            if not has_flawed_tests:
                has_flawed_tests = True
            refined_test = gen.generate_refined_test(item["prompt"], test, feedback_for_test, model)
            passed_tests.append(refined_test)
        elif "true" in feedback_for_test_binary.lower():
            passed_tests.append(test)
    return passed_tests, has_flawed_tests

def format_tests(entry_point, tests):
    # Split the test string into lines
    lines = tests.split('\n')
    
    # Find lines with 'assert' and replace 'candidate' with the entry_point
    assert_statements = [line.replace('candidate', entry_point).strip() for line in lines if 'assert' in line]
    
    return assert_statements
    
def check_if_tests_are_correct(original_dataset, current_item, generated_tests, exe):
    for item in original_dataset:
        # from item["task_id"] get the number: "task_id": "HumanEval/0"
        task_id = item["task_id"].split("/")[1]
        if task_id in current_item:
            cur_func_impl = item["prompt"] + item["canonical_solution"]
            is_passing, a, b = exe.execute(cur_func_impl, generated_tests)
    print("is_passing", is_passing, a, b)

def extract_expected_output(test_string, assert_statement):
    # Find the line that contains the assert_statement
    for line in test_string.split('\n'):
        if assert_statement in line:
            # Extract the expected output using a regex
            match = re.search(r'# output: (\d+)', line)
            if match:
                return int(match.group(1))
    return None

def replace_expected_output(s, expected_output):
    # This pattern looks for '==' followed by any characters until the end of the string
    pattern = r'==.*$'
    # This is the replacement string which will include the expected_output
    replacement = f"== {expected_output}"
    # Substitute the found pattern in the string with the replacement
    new_string = re.sub(pattern, replacement, s)
    return new_string

#TODO: take passing unit test into accountp
def run_reflexion(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False
) -> None:
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)
    original_dataset = read_jsonl("./benchmarks/HumanEval.jsonl")

    print_v = make_printv(verbose)

    num_items = len(dataset)
    num_success = resume_success_count(dataset)
    for i, item in enumerate_resume(dataset, log_path):
        cur_pass = 0
        is_solved = False
        reflections = []
        implementations = []
        test_feedback = []
        cur_func_impl = ""
        while cur_pass < pass_at_k and not is_solved:
            if is_leetcode:
                tests_i = item['visible_tests']
            else:
                tests_i = gen.internal_tests(item["prompt"], model, 6, temperature=0.0)
                # tests_i = format_tests(item["entry_point"], item["test"])
            

            check_if_tests_are_correct(original_dataset, item["name"], tests_i, exe)
            # first attempt
            cur_func_impl = gen.func_impl(item["prompt"], model, "simple")
            implementations.append(cur_func_impl)
            assert isinstance(cur_func_impl, str)
            is_passing, feedback, _ = exe.execute(cur_func_impl, tests_i)
            test_feedback.append(feedback)

            # if solved, exit early
            if is_passing:
                is_passing = exe.evaluate(
                    item["entry_point"], cur_func_impl, item["test"], timeout=10)
                is_solved = is_passing
                num_success += int(is_passing)
                break
                
            
            # use self-reflection to iteratively improve the function implementation
            cur_iter = 1
            cur_feedback = feedback
            while cur_iter < max_iters:
                print("cur_iter", cur_iter)
                # get self-reflection
                reflection = gen.self_reflection(
                    cur_func_impl, cur_feedback, model)
                reflections += [reflection]

                # apply self-reflection in the next attempt
                cur_func_impl = gen.func_impl(
                    func_sig=item["prompt"],
                    model=model,
                    strategy="reflexion",
                    prev_func_impl=cur_func_impl,
                    feedback=cur_feedback,
                    self_reflection=reflection,
                )
                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)

                # check if all internal unit tests pass
                is_passing, cur_feedback, _ = exe.execute(
                    cur_func_impl, tests_i)
                test_feedback.append(cur_feedback)

                # if solved, check if it passes the real tests, exit early
                if is_passing or cur_iter == max_iters - 1:
                    is_passing = exe.evaluate(
                        item["entry_point"], cur_func_impl, item["test"], timeout=10)
                    if is_passing:
                        item["solution"] = cur_func_impl
                        is_solved = True
                        num_success += 1
                    break
               
                # extract the first failing test from cur_feedback
                passed_tests, failed_tests = categorize_tests(cur_feedback, tests_i)
                expected_output = extract_expected_output(cur_feedback, failed_tests[0])
                replaced_test = replace_expected_output(failed_tests[0], expected_output)
                to_be_compared = [failed_tests[0], replaced_test]
                correct_test = get_correct_test(to_be_compared, item, model)
                if correct_test == "second":
                    failed_tests[0] = replaced_test
                passed_tests += failed_tests
                
               
                cur_iter += 1
            cur_pass += 1

        item["is_solved"] = is_solved
        item["reflections"] = reflections
        item["implementations"] = implementations
        item["test_feedback"] = test_feedback
        item["solution"] = cur_func_impl
        write_jsonl(log_path, [item], append=True)

        print_v(
            f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')
